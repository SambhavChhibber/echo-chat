// Typed API client for the FastAPI backend.

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') || 'http://localhost:8000'

export type Source = 'LLM_CACHE' | 'LIVE_INFERENCE'

export interface Attempt {
  deployment_id: string
  label: string
  model: string
  outcome: 'failed' | 'served' | 'cache'
  error_code: string | null
  error_message: string | null
}

export interface Meta {
  source: Source
  model: string
  deployment_id: string | null
  deployment_label: string | null
  latency_ms: number
  ttft_ms: number | null
  prompt_tokens: number
  completion_tokens: number
  tokens_estimated: boolean
  attempts: Attempt[]
  context_messages: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  meta?: Meta
}

export interface Session {
  id: string
  title: string
  system_prompt: string
  created_at: string
  updated_at: string
  message_count?: number
  messages?: Message[]
}

export interface Deployment {
  id: string
  label: string
  provider: string
  model: string
  order: number
  key_hint: string
  status: 'healthy' | 'cooldown'
  cooldown_remaining_s: number
  requests: number
  successes: number
  failures: number
  last_error: string | null
}

export interface Stats {
  replies: number
  cache_hits: number
  cache_hit_rate: number
  tokens_saved: number
  tokens_billed: number
  avg_live_ms: number | null
  avg_cache_ms: number | null
  failovers: number
}

export interface AppConfig {
  mock_mode: boolean
  cache_type: string
  cooldown_seconds: number
  default_system_prompt: string
  max_message_chars: number
  fault_injection: boolean
  fault_types: Record<string, string>
}

export interface ChatError {
  code: string
  message: string
  retryable?: boolean
  attempts?: Attempt[]
  latency_ms?: number
}

export class ApiError extends Error {
  status: number
  error: ChatError
  constructor(status: number, error: ChatError) {
    super(error.message)
    this.status = status
    this.error = error
  }
}

function clientId(): string {
  const KEY = 'echo.clientId'
  try {
    let id = localStorage.getItem(KEY)
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem(KEY, id)
    }
    return id
  } catch {
    return (window as unknown as { __cid?: string }).__cid ??= crypto.randomUUID()
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', 'X-Client-Id': clientId(), ...(init.headers || {}) },
    })
  } catch {
    throw new ApiError(0, { code: 'network', message: 'Cannot reach the server. Check your connection and retry.', retryable: true })
  }
  if (res.status === 204) return undefined as T
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new ApiError(res.status, body.error ?? { code: 'http_' + res.status, message: `Request failed (${res.status})` })
  return body as T
}

export const api = {
  health: () => request<{ status: string; mock_mode: boolean }>('/api/health'),
  config: () => request<AppConfig>('/api/config'),
  pool: () => request<{ deployments: Deployment[] }>('/api/pool'),
  stats: () => request<Stats>('/api/stats'),
  sessions: () => request<{ sessions: Session[] }>('/api/sessions'),
  session: (id: string) => request<Session>(`/api/sessions/${id}`),
  createSession: (system_prompt?: string) =>
    request<Session>('/api/sessions', { method: 'POST', body: JSON.stringify({ system_prompt }) }),
  updateSession: (id: string, patch: Partial<Pick<Session, 'title' | 'system_prompt'>>) =>
    request<Session>(`/api/sessions/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  deleteSession: (id: string) => request<void>(`/api/sessions/${id}`, { method: 'DELETE' }),
}

export interface StreamHandlers {
  onDelta: (text: string) => void
  onDone: (data: { user_message: Message; message: Message; session_title: string | null }) => void
  onError: (err: ChatError) => void
}

/** POST + Server-Sent Events (EventSource can't POST, so we parse the stream ourselves). */
export async function streamMessage(
  sessionId: string,
  body: { content: string; bypass_cache?: boolean; simulate_failures?: Record<string, string> },
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/api/sessions/${sessionId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Client-Id': clientId() },
      body: JSON.stringify(body),
      signal,
    })
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
    handlers.onError({ code: 'network', message: 'Cannot reach the server. Check your connection and retry.', retryable: true })
    return
  }
  if (!res.ok || !res.body) {
    const b = await res.json().catch(() => ({}))
    handlers.onError(b.error ?? { code: 'http_' + res.status, message: `Request failed (${res.status})`, retryable: true })
    return
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finished = false
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        let event = 'message'
        let data = ''
        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7)
          else if (line.startsWith('data: ')) data += line.slice(6)
        }
        const payload = data ? JSON.parse(data) : {}
        if (event === 'delta') handlers.onDelta(payload.text)
        else if (event === 'done') { finished = true; handlers.onDone(payload) }
        else if (event === 'error') { finished = true; handlers.onError(payload) }
      }
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') return
  }
  if (!finished) handlers.onError({ code: 'stream_interrupted', message: 'The connection dropped before the reply finished. Please retry.', retryable: true })
}
