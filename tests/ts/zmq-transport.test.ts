import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Dealer } from "zeromq";
import {
  createEvent,
  createHeartbeat,
  createResponse,
  isRuntimeMessage,
  type RuntimeMessage,
} from "@dataforge/protocol";
import { ZmqRuntimeRouter, ZmqTransport, TransportError } from "@dataforge/transport-zmq";

const IDENTITY = "fake-worker";

let router: ZmqRuntimeRouter;
let dealer: Dealer;
/** The dealer's ZeroMQ routing identity (random: zeromq 6.8 cannot set one). */
let fakeIdentity = "";
let fakeLoopStopped = false;

/** A minimal DEALER peer standing in for the Python runtime. */
async function startFakeRuntime(endpoint: string) {
  router = new ZmqRuntimeRouter({ commandEndpoint: endpoint });
  await router.bind();
  dealer = new Dealer();
  await dealer.connect(endpoint);
  fakeLoopStopped = false;

  // Announce like the real Python runtime does: the first DEALER→ROUTER
  // frame establishes the route so ROUTER→DEALER sends are not dropped.
  await dealer.send([
    Buffer.from(
      JSON.stringify(createHeartbeat({ activeJobs: 0 }, { runtimeId: IDENTITY })),
    ),
  ]);
  // Discover the identity the ROUTER actually saw for this dealer.
  const deadline = Date.now() + 10_000;
  while (router.knownIdentities().length === 0) {
    if (Date.now() > deadline) throw new Error("fake runtime never reached the router");
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  fakeIdentity = router.knownIdentities()[0];

  void (async () => {
    while (!fakeLoopStopped) {
      let frames: Buffer[];
      try {
        frames = await dealer.receive();
      } catch {
        return;
      }
      const incoming = JSON.parse(frames[0].toString("utf-8")) as RuntimeMessage;
      if (!isRuntimeMessage(incoming)) continue;

      if (incoming.type === "request") {
        if (incoming.method === "runtime.silent") continue; // never reply
        if (incoming.method === "runtime.failing") {
          await dealer.send([
            Buffer.from(
              JSON.stringify(
                createResponse(incoming.messageId, undefined as never, {
                  method: incoming.method,
                }),
              ),
            ),
          ]);
          continue;
        }
        await dealer.send([
          Buffer.from(
            JSON.stringify(
              createResponse(
                incoming.messageId,
                { runtimeId: IDENTITY, echo: incoming.payload },
                { method: incoming.method },
              ),
            ),
          ),
        ]);
      }
    }
  })();

  return async () => {
    fakeLoopStopped = true;
    dealer.close();
    await router.close();
  };
}

let cleanup: () => Promise<void>;

beforeAll(async () => {
  cleanup = await startFakeRuntime("ipc:///tmp/df-ts-transport-test.sock");
});

afterAll(async () => {
  await cleanup();
});

describe("ZmqTransport over ROUTER/DEALER", () => {
  it("correlates requests with responses", async () => {
    const transport = new ZmqTransport(router, fakeIdentity);
    await transport.connect();
    await transport.waitForPeer();
    const payload = await transport.request<{ runtimeId: string }, never>(
      "runtime.health",
      { ping: 1 },
    );
    expect(payload.runtimeId).toBe(IDENTITY);
    expect((payload as unknown as { echo: { ping: number } }).echo.ping).toBe(1);
    await transport.close();
  });

  it("throws TransportError on timeout when peer never replies", async () => {
    const transport = new ZmqTransport(router, fakeIdentity);
    await transport.connect();
    await transport.waitForPeer();
    await expect(
      transport.request("runtime.silent", {}, { timeoutMs: 200 }),
    ).rejects.toMatchObject({ code: "REQUEST_TIMEOUT" } satisfies Partial<TransportError>);
    await transport.close();
  });

  it("fans unsolicited events out to subscribers", async () => {
    const transport = new ZmqTransport(router, fakeIdentity);
    await transport.connect();
    await transport.waitForPeer();

    const iterator = transport.events()[Symbol.asyncIterator]();
    const nextPromise = iterator.next();

    // Unsolicited events flow DEALER→ROUTER, as from the Python runtime.
    await dealer.send([
      Buffer.from(
        JSON.stringify(createEvent("job.started", { jobId: "job-9" }, { jobId: "job-9" })),
      ),
    ]);
    const first = await nextPromise;
    expect(first.value?.method).toBe("job.started");

    await dealer.send([
      Buffer.from(
        JSON.stringify(createHeartbeat({ activeJobs: 2 }, { runtimeId: IDENTITY })),
      ),
    ]);
    const second = await iterator.next();
    expect(second.value?.type).toBe("heartbeat");

    await transport.close();
  });

  it("reports transport health from routing state", async () => {
    const transport = new ZmqTransport(router, fakeIdentity);
    await transport.connect();
    await transport.waitForPeer();
    const health = await transport.health();
    expect(health.connected).toBe(true);
    expect(health.lastFrameAgeMs).toBeDefined();
    await transport.close();
  });
});
