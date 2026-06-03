// Browser-side WebSocket transport to the local bridge server.
//
// The browser is the WS *client*: it connects to ws://<host>:<port>/browser,
// registers its session, then services relayed op-requests and pushes live
// events (action history, connection status) back up. Includes reconnect
// with backoff so a bridge restart or transient drop self-heals.
//
// Bridge framing (browser <-> bridge), JSON text frames:
//   bridge -> browser : { kind: "request",  request }   relayed agent op
//                       { kind: "registered" }           ack after register
//                       { kind: "ping" }
//   browser -> bridge : { kind: "register", session }    on (re)connect
//                       { kind: "response", response }    op result
//                       { kind: "event",    event }       live event
//                       { kind: "pong" }

import type {
  ConnectionStatus,
  WorkspaceEvent,
  WorkspaceRequest,
  WorkspaceResponse,
} from "./protocol.js";

export interface BrowserTransportOptions {
  sessionId: string;
  mode: "workspace" | "chat";
  /** ws://host:port — defaults to the page host on TG_BRIDGE_PORT. */
  bridgeWsUrl: string;
  onRequest: (request: WorkspaceRequest) => void;
  onStatus: (status: ConnectionStatus, detail?: string) => void;
}

const MAX_BACKOFF_MS = 15_000;

export class BrowserWsTransport {
  private ws: WebSocket | null = null;
  private backoff = 500;
  private stopped = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly opts: BrowserTransportOptions) {}

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }

  sendResponse(response: WorkspaceResponse): void {
    this.send({ kind: "response", response });
  }

  sendEvent(event: WorkspaceEvent): void {
    this.send({ kind: "event", event });
  }

  private send(msg: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  private connect(): void {
    this.opts.onStatus("reconnecting", `connecting to ${this.opts.bridgeWsUrl}`);
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.opts.bridgeWsUrl);
    } catch (err) {
      this.scheduleReconnect(`connect failed: ${(err as Error).message}`);
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = 500;
      this.send({
        kind: "register",
        session: {
          sessionId: this.opts.sessionId,
          mode: this.opts.mode,
          url: window.location.href,
        },
      });
      this.opts.onStatus("connected", "bridge connected");
    };

    ws.onmessage = (ev) => {
      let msg: { kind?: string; request?: WorkspaceRequest };
      try {
        msg = JSON.parse(typeof ev.data === "string" ? ev.data : "");
      } catch {
        return;
      }
      if (msg.kind === "request" && msg.request) {
        this.opts.onRequest(msg.request);
      } else if (msg.kind === "ping") {
        this.send({ kind: "pong" });
      }
    };

    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.scheduleReconnect("bridge connection closed");
    };

    ws.onerror = () => {
      // onclose fires next and handles reconnect; just surface status.
      this.opts.onStatus("reconnecting", "bridge connection error");
    };
  }

  private scheduleReconnect(detail: string): void {
    if (this.stopped) {
      this.opts.onStatus("disconnected", detail);
      return;
    }
    this.opts.onStatus("reconnecting", `${detail} — retrying in ${Math.round(this.backoff / 100) / 10}s`);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), this.backoff);
    this.backoff = Math.min(this.backoff * 2, MAX_BACKOFF_MS);
  }
}
