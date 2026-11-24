import { Component, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  api, ApiError, streamMessage,
  type AppConfig, type Attempt, type ChatError, type Deployment, type Message, type Meta, type Session, type Stats,
} from './api'

type UiMessage = Message & { pending?: boolean; error?: ChatError; retryOf?: string; bypass?: boolean }

const fmtTokens = (n: number) => n.toLocaleString()
const ERROR_LABEL: Record<string, string> = {
  rate_limited: '429 rate limit', quota_exceeded: '429 quota', provider_error: '5xx',
  timeout: 'timeout', auth_error: 'auth', model_not_found: 'model 404', bad_request: '400',
}

/* ------------------------------------------------------------------ metadata */
function MetaBar({ meta }: { meta: Meta }) {
  const cached = meta.source === 'LLM_CACHE'
  const failed = meta.attempts.filter((a) => a.outcome === 'failed')
  return (
    <div className="meta">
      <span className={`chip source ${cached ? 'cache' : 'live'}`} title="Resolution source">{meta.source}</span>
      <span className="chip" title={meta.deployment_label ? `Served via ${meta.deployment_label}` : 'Model'}>
        <b>model</b> {meta.model}
      </span>
      <span className="chip" title={meta.ttft_ms != null ? `First token after ${meta.ttft_ms} ms` : 'Latency'}>
        <b>latency</b> {meta.latency_ms.toLocaleString()} ms
      </span>
      <span className="chip" title={cached ? 'Served from cache — these tokens were not billed' : 'Tokens for this request only'}>
        <b>tokens</b> {fmtTokens(meta.prompt_tokens)} in · {fmtTokens(meta.completion_tokens)} out
        {meta.tokens_estimated && ' (est.)'}{cached && ' · not billed'}
      </span>
      {failed.length > 0 && <AttemptTrail attempts={meta.attempts} />}
    </div>
  )
}

function AttemptTrail({ attempts }: { attempts: Attempt[] }) {
  return (
    <span className="chip trail" title="Automatic key failover for this request">
      <b>failover</b>
      {attempts.map((a, i) => (
        <span key={a.deployment_id + i} className={`hop ${a.outcome}`} title={a.error_message ?? ''}>
          {i > 0 && <span className="arrow">→</span>}
          {a.label}
          {a.outcome === 'failed' ? ` ✕ ${ERROR_LABEL[a.error_code ?? ''] ?? a.error_code}` : ' ✓'}
        </span>
      ))}
    </span>
  )
}

/* ------------------------------------------------------------------ key pool panel */
function PoolPanel({
  pool, config, faults, setFaults, stats, open, onClose,
}: {
  pool: Deployment[]; config: AppConfig | null; faults: Record<string, string>
  setFaults: (f: Record<string, string>) => void; stats: Stats | null; open: boolean; onClose: () => void
}) {
  const active = Object.values(faults).filter(Boolean).length
  return (
    <aside className={`pool ${open ? 'open' : ''}`}>
      <div className="pool-head">
        <h2>Key pool</h2>
        <button className="ghost small mobile-only" onClick={onClose}>Close</button>
      </div>
      <p className="muted">Tried top to bottom. On a 429, quota, 5xx or timeout, the next key takes over and the failing one cools down for {config?.cooldown_seconds ?? 30}s.</p>
      <ol className="keys">
        {pool.map((d) => (
          <li key={d.id} className={`key ${d.status}`}>
            <div className="key-top">
              <span className={`dot ${d.status}`} />
              <span className="key-label">{d.label}</span>
              <span className="key-hint">{d.key_hint}</span>
            </div>
            <div className="key-sub">
              {d.model} · {d.successes} ok · {d.failures} failed
              {d.status === 'cooldown' && <span className="cool"> · cooling {Math.ceil(d.cooldown_remaining_s)}s</span>}
            </div>
            {config?.fault_injection && (
              <select
                value={faults[d.id] ?? ''}
                onChange={(e) => setFaults({ ...faults, [d.id]: e.target.value })}
                aria-label={`Simulate failure on ${d.label}`}
              >
                <option value="">No simulated failure</option>
                {Object.entries(config.fault_types).map(([k, v]) => <option key={k} value={k}>Simulate {v}</option>)}
              </select>
            )}
          </li>
        ))}
      </ol>
      {active > 0 && (
        <div className="notice warn">
          Simulating failures on {active} key{active > 1 ? 's' : ''} for your next messages.{' '}
          <button className="link" onClick={() => setFaults({})}>Clear</button>
        </div>
      )}
      {stats && stats.replies > 0 && (
        <div className="stats">
          <h3>This browser</h3>
          <dl>
            <dt>Cache hit rate</dt><dd>{Math.round(stats.cache_hit_rate * 100)}% ({stats.cache_hits}/{stats.replies})</dd>
            <dt>Tokens saved by cache</dt><dd>{fmtTokens(stats.tokens_saved)}</dd>
            <dt>Avg latency live / cache</dt><dd>{stats.avg_live_ms ?? '–'} / {stats.avg_cache_ms ?? '–'} ms</dd>
            <dt>Replies after a failover</dt><dd>{stats.failovers}</dd>
          </dl>
        </div>
      )}
    </aside>
  )
}

/* ------------------------------------------------------------------ app */
export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [backend, setBackend] = useState<'connecting' | 'up' | 'down'>('connecting')
  const [sessions, setSessions] = useState<Session[]>([])
  const [current, setCurrent] = useState<Session | null>(null)
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [pool, setPool] = useState<Deployment[]>([])
  const [stats, setStats] = useState<Stats | null>(null)
  const [faults, setFaults] = useState<Record<string, string>>({})
  const [showPrompt, setShowPrompt] = useState(false)
  const [promptDraft, setPromptDraft] = useState('')
  const [sidebar, setSidebar] = useState(false)
  const [poolOpen, setPoolOpen] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const notify = useCallback((msg: string) => { setToast(msg); setTimeout(() => setToast(null), 4000) }, [])
  const refreshSide = useCallback(() => {
    api.pool().then((p) => setPool(p.deployments)).catch(() => {})
    api.stats().then(setStats).catch(() => {})
  }, [])

  // Boot: free hosting tiers sleep, so keep retrying /health and show a "waking up" state.
  useEffect(() => {
    let cancelled = false
    const boot = async (attempt = 0) => {
      try {
        const cfg = await api.config()
        if (cancelled) return
        setConfig(cfg); setBackend('up')
        const { sessions } = await api.sessions()
        setSessions(sessions)
        const saved = localStorage.getItem('echo.currentSession')
        const pick = sessions.find((s) => s.id === saved) ?? sessions[0]
        if (pick) await openSession(pick.id)
        refreshSide()
      } catch {
        if (cancelled) return
        setBackend(attempt > 12 ? 'down' : 'connecting')
        setTimeout(() => boot(attempt + 1), Math.min(2000 + attempt * 1000, 6000))
      }
    }
    boot()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Live cooldown countdown in the pool panel.
  useEffect(() => {
    if (!pool.some((d) => d.status === 'cooldown')) return
    const t = setInterval(refreshSide, 1000)
    return () => clearInterval(t)
  }, [pool, refreshSide])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  async function openSession(id: string) {
    try {
      const s = await api.session(id)
      setCurrent(s); setMessages(s.messages ?? []); setPromptDraft(s.system_prompt)
      localStorage.setItem('echo.currentSession', id)
      setSidebar(false)
    } catch (e) {
      notify((e as ApiError).message)
    }
  }

  async function newChat() {
    try {
      const s = await api.createSession(config?.default_system_prompt)
      setSessions((prev) => [s, ...prev])
      setCurrent(s); setMessages([]); setPromptDraft(s.system_prompt)
      localStorage.setItem('echo.currentSession', s.id)
      setSidebar(false)
      inputRef.current?.focus()
      return s
    } catch (e) {
      notify((e as ApiError).message)
      return null
    }
  }

  async function removeSession(id: string) {
    try {
      await api.deleteSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
      if (current?.id === id) { setCurrent(null); setMessages([]) }
    } catch (e) { notify((e as ApiError).message) }
  }

  async function savePrompt() {
    if (!current) return
    try {
      const s = await api.updateSession(current.id, { system_prompt: promptDraft })
      setCurrent({ ...current, system_prompt: s.system_prompt })
      setShowPrompt(false)
      notify('System prompt saved for this chat')
    } catch (e) { notify((e as ApiError).message) }
  }

  async function send(text: string, opts: { bypass?: boolean; replaceId?: string } = {}) {
    const content = text.trim()
    if (!content || busy) return
    const session = current ?? (await newChat())
    if (!session) return
    setBusy(true)
    const tempUser: UiMessage = { id: 'u-' + Date.now(), role: 'user', content, created_at: new Date().toISOString() }
    const tempBot: UiMessage = { id: 'a-' + Date.now(), role: 'assistant', content: '', created_at: '', pending: true }
    setMessages((prev) => [
      ...(opts.replaceId ? prev.filter((m) => m.id !== opts.replaceId && m.retryOf !== opts.replaceId) : prev),
      tempUser,
      tempBot,
    ])
    if (!opts.replaceId) setInput('')

    const simulate = Object.fromEntries(Object.entries(faults).filter(([, v]) => v))
    await streamMessage(session.id, { content, bypass_cache: opts.bypass, simulate_failures: simulate }, {
      onDelta: (t) => setMessages((prev) => prev.map((m) => (m.id === tempBot.id ? { ...m, content: m.content + t } : m))),
      onDone: ({ user_message, message, session_title }) => {
        setMessages((prev) => prev.map((m) => (m.id === tempUser.id ? user_message : m.id === tempBot.id ? message : m)))
        if (session_title) {
          setSessions((prev) => prev.map((s) => (s.id === session.id ? { ...s, title: session_title } : s)))
          setCurrent((c) => (c && c.id === session.id ? { ...c, title: session_title } : c))
        }
        setSessions((prev) => {
          const me = prev.find((s) => s.id === session.id)
          return me ? [me, ...prev.filter((s) => s.id !== session.id)] : prev
        })
      },
      onError: (err) => {
        // keep the user's text on screen so a retry is one click
        setMessages((prev) => prev.map((m) =>
          m.id === tempBot.id ? { ...m, pending: false, content: '', error: err, retryOf: tempUser.id } : m))
      },
    })
    setBusy(false)
    refreshSide()
    inputRef.current?.focus()
  }

  function retry(failed: UiMessage) {
    const user = messages.find((m) => m.id === failed.retryOf)
    if (user) {
      setMessages((prev) => prev.filter((m) => m.id !== failed.id))
      send(user.content, { replaceId: user.id })
    }
  }

  function regenerate(index: number) {
    // Re-ask the previous user message, bypassing the cache, as a new turn.
    const user = [...messages.slice(0, index)].reverse().find((m) => m.role === 'user')
    if (user) send(user.content, { bypass: true })
  }

  const lastAssistantIdx = messages.map((m) => m.role).lastIndexOf('assistant')

  if (backend !== 'up') {
    return (
      <div className="boot">
        <div className="logo">E</div>
        {backend === 'connecting'
          ? <><h1>Waking up the server…</h1><p className="muted">Free hosting sleeps when idle; this can take up to a minute.</p><div className="spinner" /></>
          : <><h1>Server unavailable</h1><p className="muted">The backend could not be reached.</p><button onClick={() => location.reload()}>Try again</button></>}
      </div>
    )
  }

  return (
    <div className="app">
      <aside className={`sidebar ${sidebar ? 'open' : ''}`}>
        <div className="brand"><div className="logo">E</div><span>Echo</span></div>
        <button className="primary block" onClick={newChat}>+ New chat</button>
        <nav className="sessions">
          {sessions.length === 0 && <p className="muted small-pad">No chats yet.</p>}
          {sessions.map((s) => (
            <div key={s.id} className={`session ${current?.id === s.id ? 'active' : ''}`} onClick={() => openSession(s.id)}>
              <span className="session-title">{s.title}</span>
              <button className="icon" title="Delete chat" onClick={(e) => { e.stopPropagation(); removeSession(s.id) }}>✕</button>
            </div>
          ))}
        </nav>
        {config?.mock_mode && <div className="notice">Mock mode: no API keys configured on the server.</div>}
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="ghost mobile-only" onClick={() => setSidebar(!sidebar)}>☰</button>
          <h1 title={current?.title}>{current?.title ?? 'New chat'}</h1>
          <div className="topbar-actions">
            <button className={`ghost ${current?.system_prompt ? 'has-prompt' : ''}`} onClick={() => setShowPrompt(!showPrompt)}
              disabled={!current} title="Set the system prompt for this chat">System prompt</button>
            <button className="ghost mobile-only-inline" onClick={() => setPoolOpen(true)}>Keys</button>
          </div>
        </header>

        {showPrompt && current && (
          <div className="prompt-editor">
            <label htmlFor="sp">System prompt for this chat</label>
            <textarea id="sp" value={promptDraft} onChange={(e) => setPromptDraft(e.target.value)} rows={3}
              placeholder="e.g. You are a senior Python reviewer. Answer in bullet points." />
            <div className="row">
              <button className="ghost" onClick={() => { setPromptDraft(current.system_prompt); setShowPrompt(false) }}>Cancel</button>
              <button className="primary" onClick={savePrompt}>Save</button>
            </div>
          </div>
        )}

        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty">
              <h2>What can I help with?</h2>
              <p className="muted">Each reply shows where it came from: cache or live model, which model answered, how long it took and how many tokens it used.</p>
              <div className="suggestions">
                {['Explain LLM response caching in 3 bullets', 'Write a haiku about failover', 'What is the capital of Australia?'].map((s) => (
                  <button key={s} className="suggestion" onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={m.id} className={`msg ${m.role}`}>
              {m.role === 'user' ? (
                <div className="bubble">{m.content}</div>
              ) : m.error ? (
                <div className="bubble error" role="alert">
                  <strong>Couldn’t get a reply.</strong> {m.error.message}
                  {m.error.attempts && m.error.attempts.length > 0 && (
                    <div className="meta"><AttemptTrail attempts={m.error.attempts} /></div>
                  )}
                  {m.error.retryable !== false && <div><button className="small" onClick={() => retry(m)} disabled={busy}>Retry</button></div>}
                </div>
              ) : (
                <div className="answer">
                  {m.pending && !m.content ? <div className="typing"><span /><span /><span /></div> : (
                    <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown></div>
                  )}
                  {m.meta && <MetaBar meta={m.meta} />}
                  {m.meta && i === lastAssistantIdx && !busy && (
                    <div className="actions">
                      <button className="link" onClick={() => navigator.clipboard?.writeText(m.content).then(() => notify('Copied'))}>Copy</button>
                      <button className="link" onClick={() => regenerate(i)} title="Ask again, skipping the cache">Regenerate (skip cache)</button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input) }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(input) } }}
            placeholder="Message Echo…  (Enter to send, Shift+Enter for a new line)"
            rows={1}
            maxLength={config?.max_message_chars}
            aria-label="Message"
          />
          <button className="primary" type="submit" disabled={busy || !input.trim()}>{busy ? '…' : 'Send'}</button>
        </form>
      </main>

      <PoolPanel pool={pool} config={config} faults={faults} setFaults={setFaults} stats={stats} open={poolOpen} onClose={() => setPoolOpen(false)} />
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ error boundary */
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div className="boot">
        <h1>Something went wrong</h1>
        <p className="muted">Your chats are saved on the server.</p>
        <button onClick={() => location.reload()}>Reload</button>
      </div>
    )
  }
}
