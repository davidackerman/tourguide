// Where is the bridge? One answer for the whole page.
//
// The bridge (bridge/server.mjs) always runs on the user's own machine and
// speaks plain ws:// / http://, never TLS. So:
//   - Page from the local preview/dev server (http://localhost:4173), or from
//     another machine on the LAN (http://10.x.y.z:4173, a share link): the
//     bridge is on the host that served the page.
//   - Page served over HTTPS from a public host (the Cloudflare Pages build):
//     the bridge can only be on THIS machine's loopback. Browsers block
//     ws://<public host> from an HTTPS page (mixed content) but exempt
//     localhost, so default to localhost.
// `?bridge=host[:port]` overrides all of it.
//
// Keep this dependency-free: main.ts and browser.ts both import it.

const DEFAULT_BRIDGE_PORT = "7723";

export interface BridgeTarget {
  host: string;
  port: string;
  /** True when `?bridge=` named the bridge explicitly. */
  explicit: boolean;
}

function isLoopbackHost(host: string): boolean {
  return (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "127.0.0.1" ||
    host === "[::1]" ||
    host === "::1"
  );
}

/** True when the page came over HTTPS from somewhere other than this machine,
 *  i.e. the hosted static build. Such a page has no bridge of its own. */
export function isHostedPage(loc: Location = window.location): boolean {
  return loc.protocol === "https:" && !isLoopbackHost(loc.hostname);
}

export function resolveBridgeTarget(loc: Location = window.location): BridgeTarget {
  const params = new URLSearchParams(loc.search);
  const port = params.get("bridgePort") || DEFAULT_BRIDGE_PORT;
  const override = params.get("bridge");
  if (override) {
    const [oHost, oPort] = override
      .replace(/^(wss?|https?):\/\//, "")
      .replace(/\/.*$/, "")
      .split(":");
    return { host: oHost || "localhost", port: oPort || port, explicit: true };
  }
  const host = isHostedPage(loc) ? "localhost" : loc.hostname || "localhost";
  return { host, port, explicit: false };
}

/** ws://host:port/browser — where the tab registers with the bridge. */
export function bridgeWsUrl(t: BridgeTarget = resolveBridgeTarget()): string {
  return `ws://${t.host}:${t.port}/browser`;
}

/** http://host:port — base for the bridge's HTTP routes (share-state, viewer-fly, …). */
export function bridgeHttpBase(t: BridgeTarget = resolveBridgeTarget()): string {
  return `http://${t.host}:${t.port}`;
}
