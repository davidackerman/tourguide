import { WebSocket } from "ws";
const PORT = process.env.TG_BRIDGE_PORT || 7723;
const ws = new WebSocket(`ws://localhost:${PORT}/browser`);
ws.on("error", () => {});
ws.on("open", () => ws.send(JSON.stringify({ kind: "register", session: { sessionId: "sim-1", mode: "workspace", url: "http://localhost:5173/?mode=workspace" } })));
ws.on("message", (d) => {
  const m = JSON.parse(d.toString());
  if (m.kind === "request") {
    const op = m.request.op;
    let result = { op };
    if (op === "get_session") result = { sessionId:"sim-1", mode:"workspace", descriptor:{id:"fibsem",name:"fibsem-uint8"}, viewer:{ layers:[{name:"fibsem-uint8",type:"image",visible:true},{name:"vesicle_seg",type:"segmentation",visible:true}], selectedSegmentsByLayer:{}, position:[24000,3199,16684] }, tables:[], plots:[], savedStates:[], recording:{active:false} };
    ws.send(JSON.stringify({ kind: "response", response: { id: m.request.id, ok: true, result } }));
  } else if (m.kind === "ping") ws.send(JSON.stringify({ kind: "pong" }));
});
