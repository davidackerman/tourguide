// HTTP transport client for the Workspace API.
//
// This is the request/response half of the API: an agent (or an in-page
// tool, or a test script) POSTs a WorkspaceRequest to the bridge's /op
// endpoint and gets a WorkspaceResponse back. The Python SDK and the MCP
// adapter mirror this exact shape — keep them in sync.
//
// (The browser session itself does NOT use this; it receives relayed
// requests over the WebSocket transport. This client is for callers that
// drive the workspace from outside.)

import type {
  ActionSource,
  SessionSummary,
  WorkspaceOp,
  WorkspaceRequest,
  WorkspaceResponse,
  WorkspaceSessionRecord,
} from "./protocol.js";

export interface BridgeHealth {
  ok: boolean;
  version: string;
  sessions: number;
}

export class WorkspaceHttpClient {
  constructor(
    private readonly baseUrl: string,
    private readonly source: ActionSource = "local_api",
  ) {}

  async health(): Promise<BridgeHealth> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`bridge health ${res.status}`);
    return (await res.json()) as BridgeHealth;
  }

  async sessions(): Promise<WorkspaceSessionRecord[]> {
    const res = await fetch(`${this.baseUrl}/sessions`);
    if (!res.ok) throw new Error(`bridge sessions ${res.status}`);
    return (await res.json()) as WorkspaceSessionRecord[];
  }

  /** Issue a single operation and return its result, throwing on error. */
  async call<R = unknown>(op: WorkspaceOp, params?: unknown): Promise<R> {
    const request: WorkspaceRequest = {
      id: cryptoId(),
      op,
      params,
      source: this.source,
    };
    const res = await fetch(`${this.baseUrl}/op`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!res.ok) throw new Error(`bridge /op ${res.status}: ${await res.text()}`);
    const envelope = (await res.json()) as WorkspaceResponse<R>;
    if (!envelope.ok) throw new Error(envelope.error?.message ?? "workspace op failed");
    return envelope.result as R;
  }

  getSession(): Promise<SessionSummary> {
    return this.call<SessionSummary>("get_session");
  }
}

function cryptoId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `req-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}
