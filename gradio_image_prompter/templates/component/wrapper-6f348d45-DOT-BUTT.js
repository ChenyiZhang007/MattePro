import E from "./__vite-browser-external-DYxpcVy9.js";
function et(s) {
  return s && s.__esModule && Object.prototype.hasOwnProperty.call(s, "default") ? s.default : s;
}
function tt(s) {
  if (s.__esModule)
    return s;
  var e = s.default;
  if (typeof e == "function") {
    var t = function r() {
      if (this instanceof r) {
        var i = [null];
        i.push.apply(i, arguments);
        var n = Function.bind.apply(e, i);
        return new n();
      }
      return e.apply(this, arguments);
    };
    t.prototype = e.prototype;
  } else
    t = {};
  return Object.defineProperty(t, "__esModule", { value: !0 }), Object.keys(s).forEach(function(r) {
    var i = Object.getOwnPropertyDescriptor(s, r);
    Object.defineProperty(t, r, i.get ? i : {
      enumerable: !0,
      get: function() {
        return s[r];
      }
    });
  }), t;
}
const { Duplex: ps } = E;
var X = { exports: {} }, U = {
  BINARY_TYPES: ["nodebuffer", "arraybuffer", "fragments"],
  EMPTY_BUFFER: Buffer.alloc(0),
  GUID: "258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
  kForOnEventAttribute: Symbol("kIsForOnEventAttribute"),
  kListener: Symbol("kListener"),
  kStatusCode: Symbol("status-code"),
  kWebSocket: Symbol("websocket"),
  NOOP: () => {
  }
}, st, rt;
const { EMPTY_BUFFER: it } = U, ue = Buffer[Symbol.species];
function nt(s, e) {
  if (s.length === 0)
    return it;
  if (s.length === 1)
    return s[0];
  const t = Buffer.allocUnsafe(e);
  let r = 0;
  for (let i = 0; i < s.length; i++) {
    const n = s[i];
    t.set(n, r), r += n.length;
  }
  return r < e ? new ue(t.buffer, t.byteOffset, r) : t;
}
function Ae(s, e, t, r, i) {
  for (let n = 0; n < i; n++)
    t[r + n] = s[n] ^ e[n & 3];
}
function We(s, e) {
  for (let t = 0; t < s.length; t++)
    s[t] ^= e[t & 3];
}
function ot(s) {
  return s.length === s.buffer.byteLength ? s.buffer : s.buffer.slice(s.byteOffset, s.byteOffset + s.length);
}
function _e(s) {
  if (_e.readOnly = !0, Buffer.isBuffer(s))
    return s;
  let e;
  return s instanceof ArrayBuffer ? e = new ue(s) : ArrayBuffer.isView(s) ? e = new ue(s.buffer, s.byteOffset, s.byteLength) : (e = Buffer.from(s), _e.readOnly = !1), e;
}
X.exports = {
  concat: nt,
  mask: Ae,
  toArrayBuffer: ot,
  toBuffer: _e,
  unmask: We
};
if (!process.env.WS_NO_BUFFER_UTIL)
  try {
    const s = require("bufferutil");
    rt = X.exports.mask = function(e, t, r, i, n) {
      n < 48 ? Ae(e, t, r, i, n) : s.mask(e, t, r, i, n);
    }, st = X.exports.unmask = function(e, t) {
      e.length < 32 ? We(e, t) : s.unmask(e, t);
    };
  } catch {
  }
var Q = X.exports;
const Se = Symbol("kDone"), ie = Symbol("kRun");
let at = class {
  /**
   * Creates a new `Limiter`.
   *
   * @param {Number} [concurrency=Infinity] The maximum number of jobs allowed
   *     to run concurrently
   */
  constructor(e) {
    this[Se] = () => {
      this.pending--, this[ie]();
    }, this.concurrency = e || 1 / 0, this.jobs = [], this.pending = 0;
  }
  /**
   * Adds a job to the queue.
   *
   * @param {Function} job The job to run
   * @public
   */
  add(e) {
    this.jobs.push(e), this[ie]();
  }
  /**
   * Removes a job from the queue and runs it if possible.
   *
   * @private
   */
  [ie]() {
    if (this.pending !== this.concurrency && this.jobs.length) {
      const e = this.jobs.shift();
      this.pending++, e(this[Se]);
    }
  }
};
var ft = at;
const $ = E, Ee = Q, lt = ft, { kStatusCode: je } = U, ht = Buffer[Symbol.species], ct = Buffer.from([0, 0, 255, 255]), K = Symbol("permessage-deflate"), k = Symbol("total-length"), W = Symbol("callback"), T = Symbol("buffers"), z = Symbol("error");
let G, ut = class {
  /**
   * Creates a PerMessageDeflate instance.
   *
   * @param {Object} [options] Configuration options
   * @param {(Boolean|Number)} [options.clientMaxWindowBits] Advertise support
   *     for, or request, a custom client window size
   * @param {Boolean} [options.clientNoContextTakeover=false] Advertise/
   *     acknowledge disabling of client context takeover
   * @param {Number} [options.concurrencyLimit=10] The number of concurrent
   *     calls to zlib
   * @param {(Boolean|Number)} [options.serverMaxWindowBits] Request/confirm the
   *     use of a custom server window size
   * @param {Boolean} [options.serverNoContextTakeover=false] Request/accept
   *     disabling of server context takeover
   * @param {Number} [options.threshold=1024] Size (in bytes) below which
   *     messages should not be compressed if context takeover is disabled
   * @param {Object} [options.zlibDeflateOptions] Options to pass to zlib on
   *     deflate
   * @param {Object} [options.zlibInflateOptions] Options to pass to zlib on
   *     inflate
   * @param {Boolean} [isServer=false] Create the instance in either server or
   *     client mode
   * @param {Number} [maxPayload=0] The maximum allowed message length
   */
  constructor(e, t, r) {
    if (this._maxPayload = r | 0, this._options = e || {}, this._threshold = this._options.threshold !== void 0 ? this._options.threshold : 1024, this._isServer = !!t, this._deflate = null, this._inflate = null, this.params = null, !G) {
      const i = this._options.concurrencyLimit !== void 0 ? this._options.concurrencyLimit : 10;
      G = new lt(i);
    }
  }
  /**
   * @type {String}
   */
  static get extensionName() {
    return "permessage-deflate";
  }
  /**
   * Create an extension negotiation offer.
   *
   * @return {Object} Extension parameters
   * @public
   */
  offer() {
    const e = {};
    return this._options.serverNoContextTakeover && (e.server_no_context_takeover = !0), this._options.clientNoContextTakeover && (e.client_no_context_takeover = !0), this._options.serverMaxWindowBits && (e.server_max_window_bits = this._options.serverMaxWindowBits), this._options.clientMaxWindowBits ? e.client_max_window_bits = this._options.clientMaxWindowBits : this._options.clientMaxWindowBits == null && (e.client_max_window_bits = !0), e;
  }
  /**
   * Accept an extension negotiation offer/response.
   *
   * @param {Array} configurations The extension negotiation offers/reponse
   * @return {Object} Accepted configuration
   * @public
   */
  accept(e) {
    return e = this.normalizeParams(e), this.params = this._isServer ? this.acceptAsServer(e) : this.acceptAsClient(e), this.params;
  }
  /**
   * Releases all resources used by the extension.
   *
   * @public
   */
  cleanup() {
    if (this._inflate && (this._inflate.close(), this._inflate = null), this._deflate) {
      const e = this._deflate[W];
      this._deflate.close(), this._deflate = null, e && e(
        new Error(
          "The deflate stream was closed while data was being processed"
        )
      );
    }
  }
  /**
   *  Accept an extension negotiation offer.
   *
   * @param {Array} offers The extension negotiation offers
   * @return {Object} Accepted configuration
   * @private
   */
  acceptAsServer(e) {
    const t = this._options, r = e.find((i) => !(t.serverNoContextTakeover === !1 && i.server_no_context_takeover || i.server_max_window_bits && (t.serverMaxWindowBits === !1 || typeof t.serverMaxWindowBits == "number" && t.serverMaxWindowBits > i.server_max_window_bits) || typeof t.clientMaxWindowBits == "number" && !i.client_max_window_bits));
    if (!r)
      throw new Error("None of the extension offers can be accepted");
    return t.serverNoContextTakeover && (r.server_no_context_takeover = !0), t.clientNoContextTakeover && (r.client_no_context_takeover = !0), typeof t.serverMaxWindowBits == "number" && (r.server_max_window_bits = t.serverMaxWindowBits), typeof t.clientMaxWindowBits == "number" ? r.client_max_window_bits = t.clientMaxWindowBits : (r.client_max_window_bits === !0 || t.clientMaxWindowBits === !1) && delete r.client_max_window_bits, r;
  }
  /**
   * Accept the extension negotiation response.
   *
   * @param {Array} response The extension negotiation response
   * @return {Object} Accepted configuration
   * @private
   */
  acceptAsClient(e) {
    const t = e[0];
    if (this._options.clientNoContextTakeover === !1 && t.client_no_context_takeover)
      throw new Error('Unexpected parameter "client_no_context_takeover"');
    if (!t.client_max_window_bits)
      typeof this._options.clientMaxWindowBits == "number" && (t.client_max_window_bits = this._options.clientMaxWindowBits);
    else if (this._options.clientMaxWindowBits === !1 || typeof this._options.clientMaxWindowBits == "number" && t.client_max_window_bits > this._options.clientMaxWindowBits)
      throw new Error(
        'Unexpected or invalid parameter "client_max_window_bits"'
      );
    return t;
  }
  /**
   * Normalize parameters.
   *
   * @param {Array} configurations The extension negotiation offers/reponse
   * @return {Array} The offers/response with normalized parameters
   * @private
   */
  normalizeParams(e) {
    return e.forEach((t) => {
      Object.keys(t).forEach((r) => {
        let i = t[r];
        if (i.length > 1)
          throw new Error(`Parameter "${r}" must have only a single value`);
        if (i = i[0], r === "client_max_window_bits") {
          if (i !== !0) {
            const n = +i;
            if (!Number.isInteger(n) || n < 8 || n > 15)
              throw new TypeError(
                `Invalid value for parameter "${r}": ${i}`
              );
            i = n;
          } else if (!this._isServer)
            throw new TypeError(
              `Invalid value for parameter "${r}": ${i}`
            );
        } else if (r === "server_max_window_bits") {
          const n = +i;
          if (!Number.isInteger(n) || n < 8 || n > 15)
            throw new TypeError(
              `Invalid value for parameter "${r}": ${i}`
            );
          i = n;
        } else if (r === "client_no_context_takeover" || r === "server_no_context_takeover") {
          if (i !== !0)
            throw new TypeError(
              `Invalid value for parameter "${r}": ${i}`
            );
        } else
          throw new Error(`Unknown parameter "${r}"`);
        t[r] = i;
      });
    }), e;
  }
  /**
   * Decompress data. Concurrency limited.
   *
   * @param {Buffer} data Compressed data
   * @param {Boolean} fin Specifies whether or not this is the last fragment
   * @param {Function} callback Callback
   * @public
   */
  decompress(e, t, r) {
    G.add((i) => {
      this._decompress(e, t, (n, o) => {
        i(), r(n, o);
      });
    });
  }
  /**
   * Compress data. Concurrency limited.
   *
   * @param {(Buffer|String)} data Data to compress
   * @param {Boolean} fin Specifies whether or not this is the last fragment
   * @param {Function} callback Callback
   * @public
   */
  compress(e, t, r) {
    G.add((i) => {
      this._compress(e, t, (n, o) => {
        i(), r(n, o);
      });
    });
  }
  /**
   * Decompress data.
   *
   * @param {Buffer} data Compressed data
   * @param {Boolean} fin Specifies whether or not this is the last fragment
   * @param {Function} callback Callback
   * @private
   */
  _decompress(e, t, r) {
    const i = this._isServer ? "client" : "server";
    if (!this._inflate) {
      const n = `${i}_max_window_bits`, o = typeof this.params[n] != "number" ? $.Z_DEFAULT_WINDOWBITS : this.params[n];
      this._inflate = $.createInflateRaw({
        ...this._options.zlibInflateOptions,
        windowBits: o
      }), this._inflate[K] = this, this._inflate[k] = 0, this._inflate[T] = [], this._inflate.on("error", dt), this._inflate.on("data", Ge);
    }
    this._inflate[W] = r, this._inflate.write(e), t && this._inflate.write(ct), this._inflate.flush(() => {
      const n = this._inflate[z];
      if (n) {
        this._inflate.close(), this._inflate = null, r(n);
        return;
      }
      const o = Ee.concat(
        this._inflate[T],
        this._inflate[k]
      );
      this._inflate._readableState.endEmitted ? (this._inflate.close(), this._inflate = null) : (this._inflate[k] = 0, this._inflate[T] = [], t && this.params[`${i}_no_context_takeover`] && this._inflate.reset()), r(null, o);
    });
  }
  /**
   * Compress data.
   *
   * @param {(Buffer|String)} data Data to compress
   * @param {Boolean} fin Specifies whether or not this is the last fragment
   * @param {Function} callback Callback
   * @private
   */
  _compress(e, t, r) {
    const i = this._isServer ? "server" : "client";
    if (!this._deflate) {
      const n = `${i}_max_window_bits`, o = typeof this.params[n] != "number" ? $.Z_DEFAULT_WINDOWBITS : this.params[n];
      this._deflate = $.createDeflateRaw({
        ...this._options.zlibDeflateOptions,
        windowBits: o
      }), this._deflate[k] = 0, this._deflate[T] = [], this._deflate.on("data", _t);
    }
    this._deflate[W] = r, this._deflate.write(e), this._deflate.flush($.Z_SYNC_FLUSH, () => {
      if (!this._deflate)
        return;
      let n = Ee.concat(
        this._deflate[T],
        this._deflate[k]
      );
      t && (n = new ht(n.buffer, n.byteOffset, n.length - 4)), this._deflate[W] = null, this._deflate[k] = 0, this._deflate[T] = [], t && this.params[`${i}_no_context_takeover`] && this._deflate.reset(), r(null, n);
    });
  }
};
var pe = ut;
function _t(s) {
  this[T].push(s), this[k] += s.length;
}
function Ge(s) {
  if (this[k] += s.length, this[K]._maxPayload < 1 || this[k] <= this[K]._maxPayload) {
    this[T].push(s);
    return;
  }
  this[z] = new RangeError("Max payload size exceeded"), this[z].code = "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH", this[z][je] = 1009, this.removeListener("data", Ge), this.reset();
}
function dt(s) {
  this[K]._inflate = null, s[je] = 1007, this[W](s);
}
var Z = { exports: {} };
const pt = {}, mt = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: pt
}, Symbol.toStringTag, { value: "Module" })), gt = /* @__PURE__ */ tt(mt);
var ve;
const { isUtf8: be } = E, yt = [
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  // 0 - 15
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  // 16 - 31
  0,
  1,
  0,
  1,
  1,
  1,
  1,
  1,
  0,
  0,
  1,
  1,
  0,
  1,
  1,
  0,
  // 32 - 47
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  0,
  0,
  0,
  0,
  0,
  0,
  // 48 - 63
  0,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  // 64 - 79
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  0,
  0,
  0,
  1,
  1,
  // 80 - 95
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  // 96 - 111
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  0,
  1,
  0,
  1,
  0
  // 112 - 127
];
function St(s) {
  return s >= 1e3 && s <= 1014 && s !== 1004 && s !== 1005 && s !== 1006 || s >= 3e3 && s <= 4999;
}
function de(s) {
  const e = s.length;
  let t = 0;
  for (; t < e; )
    if (!(s[t] & 128))
      t++;
    else if ((s[t] & 224) === 192) {
      if (t + 1 === e || (s[t + 1] & 192) !== 128 || (s[t] & 254) === 192)
        return !1;
      t += 2;
    } else if ((s[t] & 240) === 224) {
      if (t + 2 >= e || (s[t + 1] & 192) !== 128 || (s[t + 2] & 192) !== 128 || s[t] === 224 && (s[t + 1] & 224) === 128 || // Overlong
      s[t] === 237 && (s[t + 1] & 224) === 160)
        return !1;
      t += 3;
    } else if ((s[t] & 248) === 240) {
      if (t + 3 >= e || (s[t + 1] & 192) !== 128 || (s[t + 2] & 192) !== 128 || (s[t + 3] & 192) !== 128 || s[t] === 240 && (s[t + 1] & 240) === 128 || // Overlong
      s[t] === 244 && s[t + 1] > 143 || s[t] > 244)
        return !1;
      t += 4;
    } else
      return !1;
  return !0;
}
Z.exports = {
  isValidStatusCode: St,
  isValidUTF8: de,
  tokenChars: yt
};
if (be)
  ve = Z.exports.isValidUTF8 = function(s) {
    return s.length < 24 ? de(s) : be(s);
  };
else if (!process.env.WS_NO_UTF_8_VALIDATE)
  try {
    const s = gt;
    ve = Z.exports.isValidUTF8 = function(e) {
      return e.length < 32 ? de(e) : s(e);
    };
  } catch {
  }
var J = Z.exports;
const { Writable: Et } = E, xe = pe, {
  BINARY_TYPES: vt,
  EMPTY_BUFFER: we,
  kStatusCode: bt,
  kWebSocket: xt
} = U, { concat: ne, toArrayBuffer: wt, unmask: kt } = Q, { isValidStatusCode: Ot, isValidUTF8: ke } = J, V = Buffer[Symbol.species], D = 0, Oe = 1, Te = 2, Ce = 3, oe = 4, Tt = 5;
let Ct = class extends Et {
  /**
   * Creates a Receiver instance.
   *
   * @param {Object} [options] Options object
   * @param {String} [options.binaryType=nodebuffer] The type for binary data
   * @param {Object} [options.extensions] An object containing the negotiated
   *     extensions
   * @param {Boolean} [options.isServer=false] Specifies whether to operate in
   *     client or server mode
   * @param {Number} [options.maxPayload=0] The maximum allowed message length
   * @param {Boolean} [options.skipUTF8Validation=false] Specifies whether or
   *     not to skip UTF-8 validation for text and close messages
   */
  constructor(e = {}) {
    super(), this._binaryType = e.binaryType || vt[0], this._extensions = e.extensions || {}, this._isServer = !!e.isServer, this._maxPayload = e.maxPayload | 0, this._skipUTF8Validation = !!e.skipUTF8Validation, this[xt] = void 0, this._bufferedBytes = 0, this._buffers = [], this._compressed = !1, this._payloadLength = 0, this._mask = void 0, this._fragmented = 0, this._masked = !1, this._fin = !1, this._opcode = 0, this._totalPayloadLength = 0, this._messageLength = 0, this._fragments = [], this._state = D, this._loop = !1;
  }
  /**
   * Implements `Writable.prototype._write()`.
   *
   * @param {Buffer} chunk The chunk of data to write
   * @param {String} encoding The character encoding of `chunk`
   * @param {Function} cb Callback
   * @private
   */
  _write(e, t, r) {
    if (this._opcode === 8 && this._state == D)
      return r();
    this._bufferedBytes += e.length, this._buffers.push(e), this.startLoop(r);
  }
  /**
   * Consumes `n` bytes from the buffered data.
   *
   * @param {Number} n The number of bytes to consume
   * @return {Buffer} The consumed bytes
   * @private
   */
  consume(e) {
    if (this._bufferedBytes -= e, e === this._buffers[0].length)
      return this._buffers.shift();
    if (e < this._buffers[0].length) {
      const r = this._buffers[0];
      return this._buffers[0] = new V(
        r.buffer,
        r.byteOffset + e,
        r.length - e
      ), new V(r.buffer, r.byteOffset, e);
    }
    const t = Buffer.allocUnsafe(e);
    do {
      const r = this._buffers[0], i = t.length - e;
      e >= r.length ? t.set(this._buffers.shift(), i) : (t.set(new Uint8Array(r.buffer, r.byteOffset, e), i), this._buffers[0] = new V(
        r.buffer,
        r.byteOffset + e,
        r.length - e
      )), e -= r.length;
    } while (e > 0);
    return t;
  }
  /**
   * Starts the parsing loop.
   *
   * @param {Function} cb Callback
   * @private
   */
  startLoop(e) {
    let t;
    this._loop = !0;
    do
      switch (this._state) {
        case D:
          t = this.getInfo();
          break;
        case Oe:
          t = this.getPayloadLength16();
          break;
        case Te:
          t = this.getPayloadLength64();
          break;
        case Ce:
          this.getMask();
          break;
        case oe:
          t = this.getData(e);
          break;
        default:
          this._loop = !1;
          return;
      }
    while (this._loop);
    e(t);
  }
  /**
   * Reads the first two bytes of a frame.
   *
   * @return {(RangeError|undefined)} A possible error
   * @private
   */
  getInfo() {
    if (this._bufferedBytes < 2) {
      this._loop = !1;
      return;
    }
    const e = this.consume(2);
    if (e[0] & 48)
      return this._loop = !1, m(
        RangeError,
        "RSV2 and RSV3 must be clear",
        !0,
        1002,
        "WS_ERR_UNEXPECTED_RSV_2_3"
      );
    const t = (e[0] & 64) === 64;
    if (t && !this._extensions[xe.extensionName])
      return this._loop = !1, m(
        RangeError,
        "RSV1 must be clear",
        !0,
        1002,
        "WS_ERR_UNEXPECTED_RSV_1"
      );
    if (this._fin = (e[0] & 128) === 128, this._opcode = e[0] & 15, this._payloadLength = e[1] & 127, this._opcode === 0) {
      if (t)
        return this._loop = !1, m(
          RangeError,
          "RSV1 must be clear",
          !0,
          1002,
          "WS_ERR_UNEXPECTED_RSV_1"
        );
      if (!this._fragmented)
        return this._loop = !1, m(
          RangeError,
          "invalid opcode 0",
          !0,
          1002,
          "WS_ERR_INVALID_OPCODE"
        );
      this._opcode = this._fragmented;
    } else if (this._opcode === 1 || this._opcode === 2) {
      if (this._fragmented)
        return this._loop = !1, m(
          RangeError,
          `invalid opcode ${this._opcode}`,
          !0,
          1002,
          "WS_ERR_INVALID_OPCODE"
        );
      this._compressed = t;
    } else if (this._opcode > 7 && this._opcode < 11) {
      if (!this._fin)
        return this._loop = !1, m(
          RangeError,
          "FIN must be set",
          !0,
          1002,
          "WS_ERR_EXPECTED_FIN"
        );
      if (t)
        return this._loop = !1, m(
          RangeError,
          "RSV1 must be clear",
          !0,
          1002,
          "WS_ERR_UNEXPECTED_RSV_1"
        );
      if (this._payloadLength > 125 || this._opcode === 8 && this._payloadLength === 1)
        return this._loop = !1, m(
          RangeError,
          `invalid payload length ${this._payloadLength}`,
          !0,
          1002,
          "WS_ERR_INVALID_CONTROL_PAYLOAD_LENGTH"
        );
    } else
      return this._loop = !1, m(
        RangeError,
        `invalid opcode ${this._opcode}`,
        !0,
        1002,
        "WS_ERR_INVALID_OPCODE"
      );
    if (!this._fin && !this._fragmented && (this._fragmented = this._opcode), this._masked = (e[1] & 128) === 128, this._isServer) {
      if (!this._masked)
        return this._loop = !1, m(
          RangeError,
          "MASK must be set",
          !0,
          1002,
          "WS_ERR_EXPECTED_MASK"
        );
    } else if (this._masked)
      return this._loop = !1, m(
        RangeError,
        "MASK must be clear",
        !0,
        1002,
        "WS_ERR_UNEXPECTED_MASK"
      );
    if (this._payloadLength === 126)
      this._state = Oe;
    else if (this._payloadLength === 127)
      this._state = Te;
    else
      return this.haveLength();
  }
  /**
   * Gets extended payload length (7+16).
   *
   * @return {(RangeError|undefined)} A possible error
   * @private
   */
  getPayloadLength16() {
    if (this._bufferedBytes < 2) {
      this._loop = !1;
      return;
    }
    return this._payloadLength = this.consume(2).readUInt16BE(0), this.haveLength();
  }
  /**
   * Gets extended payload length (7+64).
   *
   * @return {(RangeError|undefined)} A possible error
   * @private
   */
  getPayloadLength64() {
    if (this._bufferedBytes < 8) {
      this._loop = !1;
      return;
    }
    const e = this.consume(8), t = e.readUInt32BE(0);
    return t > Math.pow(2, 21) - 1 ? (this._loop = !1, m(
      RangeError,
      "Unsupported WebSocket frame: payload length > 2^53 - 1",
      !1,
      1009,
      "WS_ERR_UNSUPPORTED_DATA_PAYLOAD_LENGTH"
    )) : (this._payloadLength = t * Math.pow(2, 32) + e.readUInt32BE(4), this.haveLength());
  }
  /**
   * Payload length has been read.
   *
   * @return {(RangeError|undefined)} A possible error
   * @private
   */
  haveLength() {
    if (this._payloadLength && this._opcode < 8 && (this._totalPayloadLength += this._payloadLength, this._totalPayloadLength > this._maxPayload && this._maxPayload > 0))
      return this._loop = !1, m(
        RangeError,
        "Max payload size exceeded",
        !1,
        1009,
        "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH"
      );
    this._masked ? this._state = Ce : this._state = oe;
  }
  /**
   * Reads mask bytes.
   *
   * @private
   */
  getMask() {
    if (this._bufferedBytes < 4) {
      this._loop = !1;
      return;
    }
    this._mask = this.consume(4), this._state = oe;
  }
  /**
   * Reads data bytes.
   *
   * @param {Function} cb Callback
   * @return {(Error|RangeError|undefined)} A possible error
   * @private
   */
  getData(e) {
    let t = we;
    if (this._payloadLength) {
      if (this._bufferedBytes < this._payloadLength) {
        this._loop = !1;
        return;
      }
      t = this.consume(this._payloadLength), this._masked && this._mask[0] | this._mask[1] | this._mask[2] | this._mask[3] && kt(t, this._mask);
    }
    if (this._opcode > 7)
      return this.controlMessage(t);
    if (this._compressed) {
      this._state = Tt, this.decompress(t, e);
      return;
    }
    return t.length && (this._messageLength = this._totalPayloadLength, this._fragments.push(t)), this.dataMessage();
  }
  /**
   * Decompresses data.
   *
   * @param {Buffer} data Compressed data
   * @param {Function} cb Callback
   * @private
   */
  decompress(e, t) {
    this._extensions[xe.extensionName].decompress(e, this._fin, (i, n) => {
      if (i)
        return t(i);
      if (n.length) {
        if (this._messageLength += n.length, this._messageLength > this._maxPayload && this._maxPayload > 0)
          return t(
            m(
              RangeError,
              "Max payload size exceeded",
              !1,
              1009,
              "WS_ERR_UNSUPPORTED_MESSAGE_LENGTH"
            )
          );
        this._fragments.push(n);
      }
      const o = this.dataMessage();
      if (o)
        return t(o);
      this.startLoop(t);
    });
  }
  /**
   * Handles a data message.
   *
   * @return {(Error|undefined)} A possible error
   * @private
   */
  dataMessage() {
    if (this._fin) {
      const e = this._messageLength, t = this._fragments;
      if (this._totalPayloadLength = 0, this._messageLength = 0, this._fragmented = 0, this._fragments = [], this._opcode === 2) {
        let r;
        this._binaryType === "nodebuffer" ? r = ne(t, e) : this._binaryType === "arraybuffer" ? r = wt(ne(t, e)) : r = t, this.emit("message", r, !0);
      } else {
        const r = ne(t, e);
        if (!this._skipUTF8Validation && !ke(r))
          return this._loop = !1, m(
            Error,
            "invalid UTF-8 sequence",
            !0,
            1007,
            "WS_ERR_INVALID_UTF8"
          );
        this.emit("message", r, !1);
      }
    }
    this._state = D;
  }
  /**
   * Handles a control message.
   *
   * @param {Buffer} data Data to handle
   * @return {(Error|RangeError|undefined)} A possible error
   * @private
   */
  controlMessage(e) {
    if (this._opcode === 8)
      if (this._loop = !1, e.length === 0)
        this.emit("conclude", 1005, we), this.end();
      else {
        const t = e.readUInt16BE(0);
        if (!Ot(t))
          return m(
            RangeError,
            `invalid status code ${t}`,
            !0,
            1002,
            "WS_ERR_INVALID_CLOSE_CODE"
          );
        const r = new V(
          e.buffer,
          e.byteOffset + 2,
          e.length - 2
        );
        if (!this._skipUTF8Validation && !ke(r))
          return m(
            Error,
            "invalid UTF-8 sequence",
            !0,
            1007,
            "WS_ERR_INVALID_UTF8"
          );
        this.emit("conclude", t, r), this.end();
      }
    else this._opcode === 9 ? this.emit("ping", e) : this.emit("pong", e);
    this._state = D;
  }
};
var Nt = Ct;
function m(s, e, t, r, i) {
  const n = new s(
    t ? `Invalid WebSocket frame: ${e}` : e
  );
  return Error.captureStackTrace(n, m), n.code = i, n[bt] = r, n;
}
const { randomFillSync: Lt } = E, Ne = pe, { EMPTY_BUFFER: Pt } = U, { isValidStatusCode: Rt } = J, { mask: Le, toBuffer: R } = Q, b = Symbol("kByteLength"), Bt = Buffer.alloc(4);
let Ut = class L {
  /**
   * Creates a Sender instance.
   *
   * @param {(net.Socket|tls.Socket)} socket The connection socket
   * @param {Object} [extensions] An object containing the negotiated extensions
   * @param {Function} [generateMask] The function used to generate the masking
   *     key
   */
  constructor(e, t, r) {
    this._extensions = t || {}, r && (this._generateMask = r, this._maskBuffer = Buffer.alloc(4)), this._socket = e, this._firstFragment = !0, this._compress = !1, this._bufferedBytes = 0, this._deflating = !1, this._queue = [];
  }
  /**
   * Frames a piece of data according to the HyBi WebSocket protocol.
   *
   * @param {(Buffer|String)} data The data to frame
   * @param {Object} options Options object
   * @param {Boolean} [options.fin=false] Specifies whether or not to set the
   *     FIN bit
   * @param {Function} [options.generateMask] The function used to generate the
   *     masking key
   * @param {Boolean} [options.mask=false] Specifies whether or not to mask
   *     `data`
   * @param {Buffer} [options.maskBuffer] The buffer used to store the masking
   *     key
   * @param {Number} options.opcode The opcode
   * @param {Boolean} [options.readOnly=false] Specifies whether `data` can be
   *     modified
   * @param {Boolean} [options.rsv1=false] Specifies whether or not to set the
   *     RSV1 bit
   * @return {(Buffer|String)[]} The framed data
   * @public
   */
  static frame(e, t) {
    let r, i = !1, n = 2, o = !1;
    t.mask && (r = t.maskBuffer || Bt, t.generateMask ? t.generateMask(r) : Lt(r, 0, 4), o = (r[0] | r[1] | r[2] | r[3]) === 0, n = 6);
    let l;
    typeof e == "string" ? (!t.mask || o) && t[b] !== void 0 ? l = t[b] : (e = Buffer.from(e), l = e.length) : (l = e.length, i = t.mask && t.readOnly && !o);
    let f = l;
    l >= 65536 ? (n += 8, f = 127) : l > 125 && (n += 2, f = 126);
    const a = Buffer.allocUnsafe(i ? l + n : n);
    return a[0] = t.fin ? t.opcode | 128 : t.opcode, t.rsv1 && (a[0] |= 64), a[1] = f, f === 126 ? a.writeUInt16BE(l, 2) : f === 127 && (a[2] = a[3] = 0, a.writeUIntBE(l, 4, 6)), t.mask ? (a[1] |= 128, a[n - 4] = r[0], a[n - 3] = r[1], a[n - 2] = r[2], a[n - 1] = r[3], o ? [a, e] : i ? (Le(e, r, a, n, l), [a]) : (Le(e, r, e, 0, l), [a, e])) : [a, e];
  }
  /**
   * Sends a close message to the other peer.
   *
   * @param {Number} [code] The status code component of the body
   * @param {(String|Buffer)} [data] The message component of the body
   * @param {Boolean} [mask=false] Specifies whether or not to mask the message
   * @param {Function} [cb] Callback
   * @public
   */
  close(e, t, r, i) {
    let n;
    if (e === void 0)
      n = Pt;
    else {
      if (typeof e != "number" || !Rt(e))
        throw new TypeError("First argument must be a valid error code number");
      if (t === void 0 || !t.length)
        n = Buffer.allocUnsafe(2), n.writeUInt16BE(e, 0);
      else {
        const l = Buffer.byteLength(t);
        if (l > 123)
          throw new RangeError("The message must not be greater than 123 bytes");
        n = Buffer.allocUnsafe(2 + l), n.writeUInt16BE(e, 0), typeof t == "string" ? n.write(t, 2) : n.set(t, 2);
      }
    }
    const o = {
      [b]: n.length,
      fin: !0,
      generateMask: this._generateMask,
      mask: r,
      maskBuffer: this._maskBuffer,
      opcode: 8,
      readOnly: !1,
      rsv1: !1
    };
    this._deflating ? this.enqueue([this.dispatch, n, !1, o, i]) : this.sendFrame(L.frame(n, o), i);
  }
  /**
   * Sends a ping message to the other peer.
   *
   * @param {*} data The message to send
   * @param {Boolean} [mask=false] Specifies whether or not to mask `data`
   * @param {Function} [cb] Callback
   * @public
   */
  ping(e, t, r) {
    let i, n;
    if (typeof e == "string" ? (i = Buffer.byteLength(e), n = !1) : (e = R(e), i = e.length, n = R.readOnly), i > 125)
      throw new RangeError("The data size must not be greater than 125 bytes");
    const o = {
      [b]: i,
      fin: !0,
      generateMask: this._generateMask,
      mask: t,
      maskBuffer: this._maskBuffer,
      opcode: 9,
      readOnly: n,
      rsv1: !1
    };
    this._deflating ? this.enqueue([this.dispatch, e, !1, o, r]) : this.sendFrame(L.frame(e, o), r);
  }
  /**
   * Sends a pong message to the other peer.
   *
   * @param {*} data The message to send
   * @param {Boolean} [mask=false] Specifies whether or not to mask `data`
   * @param {Function} [cb] Callback
   * @public
   */
  pong(e, t, r) {
    let i, n;
    if (typeof e == "string" ? (i = Buffer.byteLength(e), n = !1) : (e = R(e), i = e.length, n = R.readOnly), i > 125)
      throw new RangeError("The data size must not be greater than 125 bytes");
    const o = {
      [b]: i,
      fin: !0,
      generateMask: this._generateMask,
      mask: t,
      maskBuffer: this._maskBuffer,
      opcode: 10,
      readOnly: n,
      rsv1: !1
    };
    this._deflating ? this.enqueue([this.dispatch, e, !1, o, r]) : this.sendFrame(L.frame(e, o), r);
  }
  /**
   * Sends a data message to the other peer.
   *
   * @param {*} data The message to send
   * @param {Object} options Options object
   * @param {Boolean} [options.binary=false] Specifies whether `data` is binary
   *     or text
   * @param {Boolean} [options.compress=false] Specifies whether or not to
   *     compress `data`
   * @param {Boolean} [options.fin=false] Specifies whether the fragment is the
   *     last one
   * @param {Boolean} [options.mask=false] Specifies whether or not to mask
   *     `data`
   * @param {Function} [cb] Callback
   * @public
   */
  send(e, t, r) {
    const i = this._extensions[Ne.extensionName];
    let n = t.binary ? 2 : 1, o = t.compress, l, f;
    if (typeof e == "string" ? (l = Buffer.byteLength(e), f = !1) : (e = R(e), l = e.length, f = R.readOnly), this._firstFragment ? (this._firstFragment = !1, o && i && i.params[i._isServer ? "server_no_context_takeover" : "client_no_context_takeover"] && (o = l >= i._threshold), this._compress = o) : (o = !1, n = 0), t.fin && (this._firstFragment = !0), i) {
      const a = {
        [b]: l,
        fin: t.fin,
        generateMask: this._generateMask,
        mask: t.mask,
        maskBuffer: this._maskBuffer,
        opcode: n,
        readOnly: f,
        rsv1: o
      };
      this._deflating ? this.enqueue([this.dispatch, e, this._compress, a, r]) : this.dispatch(e, this._compress, a, r);
    } else
      this.sendFrame(
        L.frame(e, {
          [b]: l,
          fin: t.fin,
          generateMask: this._generateMask,
          mask: t.mask,
          maskBuffer: this._maskBuffer,
          opcode: n,
          readOnly: f,
          rsv1: !1
        }),
        r
      );
  }
  /**
   * Dispatches a message.
   *
   * @param {(Buffer|String)} data The message to send
   * @param {Boolean} [compress=false] Specifies whether or not to compress
   *     `data`
   * @param {Object} options Options object
   * @param {Boolean} [options.fin=false] Specifies whether or not to set the
   *     FIN bit
   * @param {Function} [options.generateMask] The function used to generate the
   *     masking key
   * @param {Boolean} [options.mask=false] Specifies whether or not to mask
   *     `data`
   * @param {Buffer} [options.maskBuffer] The buffer used to store the masking
   *     key
   * @param {Number} options.opcode The opcode
   * @param {Boolean} [options.readOnly=false] Specifies whether `data` can be
   *     modified
   * @param {Boolean} [options.rsv1=false] Specifies whether or not to set the
   *     RSV1 bit
   * @param {Function} [cb] Callback
   * @private
   */
  dispatch(e, t, r, i) {
    if (!t) {
      this.sendFrame(L.frame(e, r), i);
      return;
    }
    const n = this._extensions[Ne.extensionName];
    this._bufferedBytes += r[b], this._deflating = !0, n.compress(e, r.fin, (o, l) => {
      if (this._socket.destroyed) {
        const f = new Error(
          "The socket was closed while data was being compressed"
        );
        typeof i == "function" && i(f);
        for (let a = 0; a < this._queue.length; a++) {
          const h = this._queue[a], c = h[h.length - 1];
          typeof c == "function" && c(f);
        }
        return;
      }
      this._bufferedBytes -= r[b], this._deflating = !1, r.readOnly = !1, this.sendFrame(L.frame(l, r), i), this.dequeue();
    });
  }
  /**
   * Executes queued send operations.
   *
   * @private
   */
  dequeue() {
    for (; !this._deflating && this._queue.length; ) {
      const e = this._queue.shift();
      this._bufferedBytes -= e[3][b], Reflect.apply(e[0], this, e.slice(1));
    }
  }
  /**
   * Enqueues a send operation.
   *
   * @param {Array} params Send operation parameters.
   * @private
   */
  enqueue(e) {
    this._bufferedBytes += e[3][b], this._queue.push(e);
  }
  /**
   * Sends a frame.
   *
   * @param {Buffer[]} list The frame to send
   * @param {Function} [cb] Callback
   * @private
   */
  sendFrame(e, t) {
    e.length === 2 ? (this._socket.cork(), this._socket.write(e[0]), this._socket.write(e[1], t), this._socket.uncork()) : this._socket.write(e[0], t);
  }
};
var It = Ut;
const { kForOnEventAttribute: F, kListener: ae } = U, Pe = Symbol("kCode"), Re = Symbol("kData"), Be = Symbol("kError"), Ue = Symbol("kMessage"), Ie = Symbol("kReason"), B = Symbol("kTarget"), Me = Symbol("kType"), $e = Symbol("kWasClean");
class I {
  /**
   * Create a new `Event`.
   *
   * @param {String} type The name of the event
   * @throws {TypeError} If the `type` argument is not specified
   */
  constructor(e) {
    this[B] = null, this[Me] = e;
  }
  /**
   * @type {*}
   */
  get target() {
    return this[B];
  }
  /**
   * @type {String}
   */
  get type() {
    return this[Me];
  }
}
Object.defineProperty(I.prototype, "target", { enumerable: !0 });
Object.defineProperty(I.prototype, "type", { enumerable: !0 });
class ee extends I {
  /**
   * Create a new `CloseEvent`.
   *
   * @param {String} type The name of the event
   * @param {Object} [options] A dictionary object that allows for setting
   *     attributes via object members of the same name
   * @param {Number} [options.code=0] The status code explaining why the
   *     connection was closed
   * @param {String} [options.reason=''] A human-readable string explaining why
   *     the connection was closed
   * @param {Boolean} [options.wasClean=false] Indicates whether or not the
   *     connection was cleanly closed
   */
  constructor(e, t = {}) {
    super(e), this[Pe] = t.code === void 0 ? 0 : t.code, this[Ie] = t.reason === void 0 ? "" : t.reason, this[$e] = t.wasClean === void 0 ? !1 : t.wasClean;
  }
  /**
   * @type {Number}
   */
  get code() {
    return this[Pe];
  }
  /**
   * @type {String}
   */
  get reason() {
    return this[Ie];
  }
  /**
   * @type {Boolean}
   */
  get wasClean() {
    return this[$e];
  }
}
Object.defineProperty(ee.prototype, "code", { enumerable: !0 });
Object.defineProperty(ee.prototype, "reason", { enumerable: !0 });
Object.defineProperty(ee.prototype, "wasClean", { enumerable: !0 });
class me extends I {
  /**
   * Create a new `ErrorEvent`.
   *
   * @param {String} type The name of the event
   * @param {Object} [options] A dictionary object that allows for setting
   *     attributes via object members of the same name
   * @param {*} [options.error=null] The error that generated this event
   * @param {String} [options.message=''] The error message
   */
  constructor(e, t = {}) {
    super(e), this[Be] = t.error === void 0 ? null : t.error, this[Ue] = t.message === void 0 ? "" : t.message;
  }
  /**
   * @type {*}
   */
  get error() {
    return this[Be];
  }
  /**
   * @type {String}
   */
  get message() {
    return this[Ue];
  }
}
Object.defineProperty(me.prototype, "error", { enumerable: !0 });
Object.defineProperty(me.prototype, "message", { enumerable: !0 });
class Ve extends I {
  /**
   * Create a new `MessageEvent`.
   *
   * @param {String} type The name of the event
   * @param {Object} [options] A dictionary object that allows for setting
   *     attributes via object members of the same name
   * @param {*} [options.data=null] The message content
   */
  constructor(e, t = {}) {
    super(e), this[Re] = t.data === void 0 ? null : t.data;
  }
  /**
   * @type {*}
   */
  get data() {
    return this[Re];
  }
}
Object.defineProperty(Ve.prototype, "data", { enumerable: !0 });
const Mt = {
  /**
   * Register an event listener.
   *
   * @param {String} type A string representing the event type to listen for
   * @param {(Function|Object)} handler The listener to add
   * @param {Object} [options] An options object specifies characteristics about
   *     the event listener
   * @param {Boolean} [options.once=false] A `Boolean` indicating that the
   *     listener should be invoked at most once after being added. If `true`,
   *     the listener would be automatically removed when invoked.
   * @public
   */
  addEventListener(s, e, t = {}) {
    for (const i of this.listeners(s))
      if (!t[F] && i[ae] === e && !i[F])
        return;
    let r;
    if (s === "message")
      r = function(n, o) {
        const l = new Ve("message", {
          data: o ? n : n.toString()
        });
        l[B] = this, q(e, this, l);
      };
    else if (s === "close")
      r = function(n, o) {
        const l = new ee("close", {
          code: n,
          reason: o.toString(),
          wasClean: this._closeFrameReceived && this._closeFrameSent
        });
        l[B] = this, q(e, this, l);
      };
    else if (s === "error")
      r = function(n) {
        const o = new me("error", {
          error: n,
          message: n.message
        });
        o[B] = this, q(e, this, o);
      };
    else if (s === "open")
      r = function() {
        const n = new I("open");
        n[B] = this, q(e, this, n);
      };
    else
      return;
    r[F] = !!t[F], r[ae] = e, t.once ? this.once(s, r) : this.on(s, r);
  },
  /**
   * Remove an event listener.
   *
   * @param {String} type A string representing the event type to remove
   * @param {(Function|Object)} handler The listener to remove
   * @public
   */
  removeEventListener(s, e) {
    for (const t of this.listeners(s))
      if (t[ae] === e && !t[F]) {
        this.removeListener(s, t);
        break;
      }
  }
};
var $t = {
  EventTarget: Mt
};
function q(s, e, t) {
  typeof s == "object" && s.handleEvent ? s.handleEvent.call(s, t) : s.call(e, t);
}
const { tokenChars: A } = J;
function w(s, e, t) {
  s[e] === void 0 ? s[e] = [t] : s[e].push(t);
}
function Dt(s) {
  const e = /* @__PURE__ */ Object.create(null);
  let t = /* @__PURE__ */ Object.create(null), r = !1, i = !1, n = !1, o, l, f = -1, a = -1, h = -1, c = 0;
  for (; c < s.length; c++)
    if (a = s.charCodeAt(c), o === void 0)
      if (h === -1 && A[a] === 1)
        f === -1 && (f = c);
      else if (c !== 0 && (a === 32 || a === 9))
        h === -1 && f !== -1 && (h = c);
      else if (a === 59 || a === 44) {
        if (f === -1)
          throw new SyntaxError(`Unexpected character at index ${c}`);
        h === -1 && (h = c);
        const v = s.slice(f, h);
        a === 44 ? (w(e, v, t), t = /* @__PURE__ */ Object.create(null)) : o = v, f = h = -1;
      } else
        throw new SyntaxError(`Unexpected character at index ${c}`);
    else if (l === void 0)
      if (h === -1 && A[a] === 1)
        f === -1 && (f = c);
      else if (a === 32 || a === 9)
        h === -1 && f !== -1 && (h = c);
      else if (a === 59 || a === 44) {
        if (f === -1)
          throw new SyntaxError(`Unexpected character at index ${c}`);
        h === -1 && (h = c), w(t, s.slice(f, h), !0), a === 44 && (w(e, o, t), t = /* @__PURE__ */ Object.create(null), o = void 0), f = h = -1;
      } else if (a === 61 && f !== -1 && h === -1)
        l = s.slice(f, c), f = h = -1;
      else
        throw new SyntaxError(`Unexpected character at index ${c}`);
    else if (i) {
      if (A[a] !== 1)
        throw new SyntaxError(`Unexpected character at index ${c}`);
      f === -1 ? f = c : r || (r = !0), i = !1;
    } else if (n)
      if (A[a] === 1)
        f === -1 && (f = c);
      else if (a === 34 && f !== -1)
        n = !1, h = c;
      else if (a === 92)
        i = !0;
      else
        throw new SyntaxError(`Unexpected character at index ${c}`);
    else if (a === 34 && s.charCodeAt(c - 1) === 61)
      n = !0;
    else if (h === -1 && A[a] === 1)
      f === -1 && (f = c);
    else if (f !== -1 && (a === 32 || a === 9))
      h === -1 && (h = c);
    else if (a === 59 || a === 44) {
      if (f === -1)
        throw new SyntaxError(`Unexpected character at index ${c}`);
      h === -1 && (h = c);
      let v = s.slice(f, h);
      r && (v = v.replace(/\\/g, ""), r = !1), w(t, l, v), a === 44 && (w(e, o, t), t = /* @__PURE__ */ Object.create(null), o = void 0), l = void 0, f = h = -1;
    } else
      throw new SyntaxError(`Unexpected character at index ${c}`);
  if (f === -1 || n || a === 32 || a === 9)
    throw new SyntaxError("Unexpected end of input");
  h === -1 && (h = c);
  const x = s.slice(f, h);
  return o === void 0 ? w(e, x, t) : (l === void 0 ? w(t, x, !0) : r ? w(t, l, x.replace(/\\/g, "")) : w(t, l, x), w(e, o, t)), e;
}
function Ft(s) {
  return Object.keys(s).map((e) => {
    let t = s[e];
    return Array.isArray(t) || (t = [t]), t.map((r) => [e].concat(
      Object.keys(r).map((i) => {
        let n = r[i];
        return Array.isArray(n) || (n = [n]), n.map((o) => o === !0 ? i : `${i}=${o}`).join("; ");
      })
    ).join("; ")).join(", ");
  }).join(", ");
}
var At = { format: Ft, parse: Dt };
const Wt = E, jt = E, Gt = E, qe = E, Vt = E, { randomBytes: qt, createHash: Yt } = E, { URL: fe } = E, C = pe, zt = Nt, Ht = It, {
  BINARY_TYPES: De,
  EMPTY_BUFFER: Y,
  GUID: Xt,
  kForOnEventAttribute: le,
  kListener: Kt,
  kStatusCode: Zt,
  kWebSocket: g,
  NOOP: Ye
} = U, {
  EventTarget: { addEventListener: Qt, removeEventListener: Jt }
} = $t, { format: es, parse: ts } = At, { toBuffer: ss } = Q, rs = 30 * 1e3, ze = Symbol("kAborted"), he = [8, 13], O = ["CONNECTING", "OPEN", "CLOSING", "CLOSED"], is = /^[!#$%&'*+\-.0-9A-Z^_`|a-z~]+$/;
let d = class _ extends Wt {
  /**
   * Create a new `WebSocket`.
   *
   * @param {(String|URL)} address The URL to which to connect
   * @param {(String|String[])} [protocols] The subprotocols
   * @param {Object} [options] Connection options
   */
  constructor(e, t, r) {
    super(), this._binaryType = De[0], this._closeCode = 1006, this._closeFrameReceived = !1, this._closeFrameSent = !1, this._closeMessage = Y, this._closeTimer = null, this._extensions = {}, this._paused = !1, this._protocol = "", this._readyState = _.CONNECTING, this._receiver = null, this._sender = null, this._socket = null, e !== null ? (this._bufferedAmount = 0, this._isServer = !1, this._redirects = 0, t === void 0 ? t = [] : Array.isArray(t) || (typeof t == "object" && t !== null ? (r = t, t = []) : t = [t]), He(this, e, t, r)) : this._isServer = !0;
  }
  /**
   * This deviates from the WHATWG interface since ws doesn't support the
   * required default "blob" type (instead we define a custom "nodebuffer"
   * type).
   *
   * @type {String}
   */
  get binaryType() {
    return this._binaryType;
  }
  set binaryType(e) {
    De.includes(e) && (this._binaryType = e, this._receiver && (this._receiver._binaryType = e));
  }
  /**
   * @type {Number}
   */
  get bufferedAmount() {
    return this._socket ? this._socket._writableState.length + this._sender._bufferedBytes : this._bufferedAmount;
  }
  /**
   * @type {String}
   */
  get extensions() {
    return Object.keys(this._extensions).join();
  }
  /**
   * @type {Boolean}
   */
  get isPaused() {
    return this._paused;
  }
  /**
   * @type {Function}
   */
  /* istanbul ignore next */
  get onclose() {
    return null;
  }
  /**
   * @type {Function}
   */
  /* istanbul ignore next */
  get onerror() {
    return null;
  }
  /**
   * @type {Function}
   */
  /* istanbul ignore next */
  get onopen() {
    return null;
  }
  /**
   * @type {Function}
   */
  /* istanbul ignore next */
  get onmessage() {
    return null;
  }
  /**
   * @type {String}
   */
  get protocol() {
    return this._protocol;
  }
  /**
   * @type {Number}
   */
  get readyState() {
    return this._readyState;
  }
  /**
   * @type {String}
   */
  get url() {
    return this._url;
  }
  /**
   * Set up the socket and the internal resources.
   *
   * @param {(net.Socket|tls.Socket)} socket The network socket between the
   *     server and client
   * @param {Buffer} head The first packet of the upgraded stream
   * @param {Object} options Options object
   * @param {Function} [options.generateMask] The function used to generate the
   *     masking key
   * @param {Number} [options.maxPayload=0] The maximum allowed message size
   * @param {Boolean} [options.skipUTF8Validation=false] Specifies whether or
   *     not to skip UTF-8 validation for text and close messages
   * @private
   */
  setSocket(e, t, r) {
    const i = new zt({
      binaryType: this.binaryType,
      extensions: this._extensions,
      isServer: this._isServer,
      maxPayload: r.maxPayload,
      skipUTF8Validation: r.skipUTF8Validation
    });
    this._sender = new Ht(e, this._extensions, r.generateMask), this._receiver = i, this._socket = e, i[g] = this, e[g] = this, i.on("conclude", fs), i.on("drain", ls), i.on("error", hs), i.on("message", cs), i.on("ping", us), i.on("pong", _s), e.setTimeout(0), e.setNoDelay(), t.length > 0 && e.unshift(t), e.on("close", Ke), e.on("data", te), e.on("end", Ze), e.on("error", Qe), this._readyState = _.OPEN, this.emit("open");
  }
  /**
   * Emit the `'close'` event.
   *
   * @private
   */
  emitClose() {
    if (!this._socket) {
      this._readyState = _.CLOSED, this.emit("close", this._closeCode, this._closeMessage);
      return;
    }
    this._extensions[C.extensionName] && this._extensions[C.extensionName].cleanup(), this._receiver.removeAllListeners(), this._readyState = _.CLOSED, this.emit("close", this._closeCode, this._closeMessage);
  }
  /**
   * Start a closing handshake.
   *
   *          +----------+   +-----------+   +----------+
   *     - - -|ws.close()|-->|close frame|-->|ws.close()|- - -
   *    |     +----------+   +-----------+   +----------+     |
   *          +----------+   +-----------+         |
   * CLOSING  |ws.close()|<--|close frame|<--+-----+       CLOSING
   *          +----------+   +-----------+   |
   *    |           |                        |   +---+        |
   *                +------------------------+-->|fin| - - - -
   *    |         +---+                      |   +---+
   *     - - - - -|fin|<---------------------+
   *              +---+
   *
   * @param {Number} [code] Status code explaining why the connection is closing
   * @param {(String|Buffer)} [data] The reason why the connection is
   *     closing
   * @public
   */
  close(e, t) {
    if (this.readyState !== _.CLOSED) {
      if (this.readyState === _.CONNECTING) {
        S(this, this._req, "WebSocket was closed before the connection was established");
        return;
      }
      if (this.readyState === _.CLOSING) {
        this._closeFrameSent && (this._closeFrameReceived || this._receiver._writableState.errorEmitted) && this._socket.end();
        return;
      }
      this._readyState = _.CLOSING, this._sender.close(e, t, !this._isServer, (r) => {
        r || (this._closeFrameSent = !0, (this._closeFrameReceived || this._receiver._writableState.errorEmitted) && this._socket.end());
      }), this._closeTimer = setTimeout(
        this._socket.destroy.bind(this._socket),
        rs
      );
    }
  }
  /**
   * Pause the socket.
   *
   * @public
   */
  pause() {
    this.readyState === _.CONNECTING || this.readyState === _.CLOSED || (this._paused = !0, this._socket.pause());
  }
  /**
   * Send a ping.
   *
   * @param {*} [data] The data to send
   * @param {Boolean} [mask] Indicates whether or not to mask `data`
   * @param {Function} [cb] Callback which is executed when the ping is sent
   * @public
   */
  ping(e, t, r) {
    if (this.readyState === _.CONNECTING)
      throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
    if (typeof e == "function" ? (r = e, e = t = void 0) : typeof t == "function" && (r = t, t = void 0), typeof e == "number" && (e = e.toString()), this.readyState !== _.OPEN) {
      ce(this, e, r);
      return;
    }
    t === void 0 && (t = !this._isServer), this._sender.ping(e || Y, t, r);
  }
  /**
   * Send a pong.
   *
   * @param {*} [data] The data to send
   * @param {Boolean} [mask] Indicates whether or not to mask `data`
   * @param {Function} [cb] Callback which is executed when the pong is sent
   * @public
   */
  pong(e, t, r) {
    if (this.readyState === _.CONNECTING)
      throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
    if (typeof e == "function" ? (r = e, e = t = void 0) : typeof t == "function" && (r = t, t = void 0), typeof e == "number" && (e = e.toString()), this.readyState !== _.OPEN) {
      ce(this, e, r);
      return;
    }
    t === void 0 && (t = !this._isServer), this._sender.pong(e || Y, t, r);
  }
  /**
   * Resume the socket.
   *
   * @public
   */
  resume() {
    this.readyState === _.CONNECTING || this.readyState === _.CLOSED || (this._paused = !1, this._receiver._writableState.needDrain || this._socket.resume());
  }
  /**
   * Send a data message.
   *
   * @param {*} data The message to send
   * @param {Object} [options] Options object
   * @param {Boolean} [options.binary] Specifies whether `data` is binary or
   *     text
   * @param {Boolean} [options.compress] Specifies whether or not to compress
   *     `data`
   * @param {Boolean} [options.fin=true] Specifies whether the fragment is the
   *     last one
   * @param {Boolean} [options.mask] Specifies whether or not to mask `data`
   * @param {Function} [cb] Callback which is executed when data is written out
   * @public
   */
  send(e, t, r) {
    if (this.readyState === _.CONNECTING)
      throw new Error("WebSocket is not open: readyState 0 (CONNECTING)");
    if (typeof t == "function" && (r = t, t = {}), typeof e == "number" && (e = e.toString()), this.readyState !== _.OPEN) {
      ce(this, e, r);
      return;
    }
    const i = {
      binary: typeof e != "string",
      mask: !this._isServer,
      compress: !0,
      fin: !0,
      ...t
    };
    this._extensions[C.extensionName] || (i.compress = !1), this._sender.send(e || Y, i, r);
  }
  /**
   * Forcibly close the connection.
   *
   * @public
   */
  terminate() {
    if (this.readyState !== _.CLOSED) {
      if (this.readyState === _.CONNECTING) {
        S(this, this._req, "WebSocket was closed before the connection was established");
        return;
      }
      this._socket && (this._readyState = _.CLOSING, this._socket.destroy());
    }
  }
};
Object.defineProperty(d, "CONNECTING", {
  enumerable: !0,
  value: O.indexOf("CONNECTING")
});
Object.defineProperty(d.prototype, "CONNECTING", {
  enumerable: !0,
  value: O.indexOf("CONNECTING")
});
Object.defineProperty(d, "OPEN", {
  enumerable: !0,
  value: O.indexOf("OPEN")
});
Object.defineProperty(d.prototype, "OPEN", {
  enumerable: !0,
  value: O.indexOf("OPEN")
});
Object.defineProperty(d, "CLOSING", {
  enumerable: !0,
  value: O.indexOf("CLOSING")
});
Object.defineProperty(d.prototype, "CLOSING", {
  enumerable: !0,
  value: O.indexOf("CLOSING")
});
Object.defineProperty(d, "CLOSED", {
  enumerable: !0,
  value: O.indexOf("CLOSED")
});
Object.defineProperty(d.prototype, "CLOSED", {
  enumerable: !0,
  value: O.indexOf("CLOSED")
});
[
  "binaryType",
  "bufferedAmount",
  "extensions",
  "isPaused",
  "protocol",
  "readyState",
  "url"
].forEach((s) => {
  Object.defineProperty(d.prototype, s, { enumerable: !0 });
});
["open", "error", "close", "message"].forEach((s) => {
  Object.defineProperty(d.prototype, `on${s}`, {
    enumerable: !0,
    get() {
      for (const e of this.listeners(s))
        if (e[le])
          return e[Kt];
      return null;
    },
    set(e) {
      for (const t of this.listeners(s))
        if (t[le]) {
          this.removeListener(s, t);
          break;
        }
      typeof e == "function" && this.addEventListener(s, e, {
        [le]: !0
      });
    }
  });
});
d.prototype.addEventListener = Qt;
d.prototype.removeEventListener = Jt;
var ns = d;
function He(s, e, t, r) {
  const i = {
    protocolVersion: he[1],
    maxPayload: 104857600,
    skipUTF8Validation: !1,
    perMessageDeflate: !0,
    followRedirects: !1,
    maxRedirects: 10,
    ...r,
    createConnection: void 0,
    socketPath: void 0,
    hostname: void 0,
    protocol: void 0,
    timeout: void 0,
    method: "GET",
    host: void 0,
    path: void 0,
    port: void 0
  };
  if (!he.includes(i.protocolVersion))
    throw new RangeError(
      `Unsupported protocol version: ${i.protocolVersion} (supported versions: ${he.join(", ")})`
    );
  let n;
  if (e instanceof fe)
    n = e, s._url = e.href;
  else {
    try {
      n = new fe(e);
    } catch {
      throw new SyntaxError(`Invalid URL: ${e}`);
    }
    s._url = e;
  }
  const o = n.protocol === "wss:", l = n.protocol === "ws+unix:";
  let f;
  if (n.protocol !== "ws:" && !o && !l ? f = `The URL's protocol must be one of "ws:", "wss:", or "ws+unix:"` : l && !n.pathname ? f = "The URL's pathname is empty" : n.hash && (f = "The URL contains a fragment identifier"), f) {
    const u = new SyntaxError(f);
    if (s._redirects === 0)
      throw u;
    H(s, u);
    return;
  }
  const a = o ? 443 : 80, h = qt(16).toString("base64"), c = o ? jt.request : Gt.request, x = /* @__PURE__ */ new Set();
  let v;
  if (i.createConnection = o ? as : os, i.defaultPort = i.defaultPort || a, i.port = n.port || a, i.host = n.hostname.startsWith("[") ? n.hostname.slice(1, -1) : n.hostname, i.headers = {
    ...i.headers,
    "Sec-WebSocket-Version": i.protocolVersion,
    "Sec-WebSocket-Key": h,
    Connection: "Upgrade",
    Upgrade: "websocket"
  }, i.path = n.pathname + n.search, i.timeout = i.handshakeTimeout, i.perMessageDeflate && (v = new C(
    i.perMessageDeflate !== !0 ? i.perMessageDeflate : {},
    !1,
    i.maxPayload
  ), i.headers["Sec-WebSocket-Extensions"] = es({
    [C.extensionName]: v.offer()
  })), t.length) {
    for (const u of t) {
      if (typeof u != "string" || !is.test(u) || x.has(u))
        throw new SyntaxError(
          "An invalid or duplicated subprotocol was specified"
        );
      x.add(u);
    }
    i.headers["Sec-WebSocket-Protocol"] = t.join(",");
  }
  if (i.origin && (i.protocolVersion < 13 ? i.headers["Sec-WebSocket-Origin"] = i.origin : i.headers.Origin = i.origin), (n.username || n.password) && (i.auth = `${n.username}:${n.password}`), l) {
    const u = i.path.split(":");
    i.socketPath = u[0], i.path = u[1];
  }
  let p;
  if (i.followRedirects) {
    if (s._redirects === 0) {
      s._originalIpc = l, s._originalSecure = o, s._originalHostOrSocketPath = l ? i.socketPath : n.host;
      const u = r && r.headers;
      if (r = { ...r, headers: {} }, u)
        for (const [y, P] of Object.entries(u))
          r.headers[y.toLowerCase()] = P;
    } else if (s.listenerCount("redirect") === 0) {
      const u = l ? s._originalIpc ? i.socketPath === s._originalHostOrSocketPath : !1 : s._originalIpc ? !1 : n.host === s._originalHostOrSocketPath;
      (!u || s._originalSecure && !o) && (delete i.headers.authorization, delete i.headers.cookie, u || delete i.headers.host, i.auth = void 0);
    }
    i.auth && !r.headers.authorization && (r.headers.authorization = "Basic " + Buffer.from(i.auth).toString("base64")), p = s._req = c(i), s._redirects && s.emit("redirect", s.url, p);
  } else
    p = s._req = c(i);
  i.timeout && p.on("timeout", () => {
    S(s, p, "Opening handshake has timed out");
  }), p.on("error", (u) => {
    p === null || p[ze] || (p = s._req = null, H(s, u));
  }), p.on("response", (u) => {
    const y = u.headers.location, P = u.statusCode;
    if (y && i.followRedirects && P >= 300 && P < 400) {
      if (++s._redirects > i.maxRedirects) {
        S(s, p, "Maximum redirects exceeded");
        return;
      }
      p.abort();
      let j;
      try {
        j = new fe(y, e);
      } catch {
        const N = new SyntaxError(`Invalid URL: ${y}`);
        H(s, N);
        return;
      }
      He(s, j, t, r);
    } else s.emit("unexpected-response", p, u) || S(
      s,
      p,
      `Unexpected server response: ${u.statusCode}`
    );
  }), p.on("upgrade", (u, y, P) => {
    if (s.emit("upgrade", u), s.readyState !== d.CONNECTING)
      return;
    if (p = s._req = null, u.headers.upgrade.toLowerCase() !== "websocket") {
      S(s, y, "Invalid Upgrade header");
      return;
    }
    const j = Yt("sha1").update(h + Xt).digest("base64");
    if (u.headers["sec-websocket-accept"] !== j) {
      S(s, y, "Invalid Sec-WebSocket-Accept header");
      return;
    }
    const M = u.headers["sec-websocket-protocol"];
    let N;
    if (M !== void 0 ? x.size ? x.has(M) || (N = "Server sent an invalid subprotocol") : N = "Server sent a subprotocol but none was requested" : x.size && (N = "Server sent no subprotocol"), N) {
      S(s, y, N);
      return;
    }
    M && (s._protocol = M);
    const ge = u.headers["sec-websocket-extensions"];
    if (ge !== void 0) {
      if (!v) {
        S(s, y, "Server sent a Sec-WebSocket-Extensions header but no extension was requested");
        return;
      }
      let se;
      try {
        se = ts(ge);
      } catch {
        S(s, y, "Invalid Sec-WebSocket-Extensions header");
        return;
      }
      const ye = Object.keys(se);
      if (ye.length !== 1 || ye[0] !== C.extensionName) {
        S(s, y, "Server indicated an extension that was not requested");
        return;
      }
      try {
        v.accept(se[C.extensionName]);
      } catch {
        S(s, y, "Invalid Sec-WebSocket-Extensions header");
        return;
      }
      s._extensions[C.extensionName] = v;
    }
    s.setSocket(y, P, {
      generateMask: i.generateMask,
      maxPayload: i.maxPayload,
      skipUTF8Validation: i.skipUTF8Validation
    });
  }), i.finishRequest ? i.finishRequest(p, s) : p.end();
}
function H(s, e) {
  s._readyState = d.CLOSING, s.emit("error", e), s.emitClose();
}
function os(s) {
  return s.path = s.socketPath, qe.connect(s);
}
function as(s) {
  return s.path = void 0, !s.servername && s.servername !== "" && (s.servername = qe.isIP(s.host) ? "" : s.host), Vt.connect(s);
}
function S(s, e, t) {
  s._readyState = d.CLOSING;
  const r = new Error(t);
  Error.captureStackTrace(r, S), e.setHeader ? (e[ze] = !0, e.abort(), e.socket && !e.socket.destroyed && e.socket.destroy(), process.nextTick(H, s, r)) : (e.destroy(r), e.once("error", s.emit.bind(s, "error")), e.once("close", s.emitClose.bind(s)));
}
function ce(s, e, t) {
  if (e) {
    const r = ss(e).length;
    s._socket ? s._sender._bufferedBytes += r : s._bufferedAmount += r;
  }
  if (t) {
    const r = new Error(
      `WebSocket is not open: readyState ${s.readyState} (${O[s.readyState]})`
    );
    process.nextTick(t, r);
  }
}
function fs(s, e) {
  const t = this[g];
  t._closeFrameReceived = !0, t._closeMessage = e, t._closeCode = s, t._socket[g] !== void 0 && (t._socket.removeListener("data", te), process.nextTick(Xe, t._socket), s === 1005 ? t.close() : t.close(s, e));
}
function ls() {
  const s = this[g];
  s.isPaused || s._socket.resume();
}
function hs(s) {
  const e = this[g];
  e._socket[g] !== void 0 && (e._socket.removeListener("data", te), process.nextTick(Xe, e._socket), e.close(s[Zt])), e.emit("error", s);
}
function Fe() {
  this[g].emitClose();
}
function cs(s, e) {
  this[g].emit("message", s, e);
}
function us(s) {
  const e = this[g];
  e.pong(s, !e._isServer, Ye), e.emit("ping", s);
}
function _s(s) {
  this[g].emit("pong", s);
}
function Xe(s) {
  s.resume();
}
function Ke() {
  const s = this[g];
  this.removeListener("close", Ke), this.removeListener("data", te), this.removeListener("end", Ze), s._readyState = d.CLOSING;
  let e;
  !this._readableState.endEmitted && !s._closeFrameReceived && !s._receiver._writableState.errorEmitted && (e = s._socket.read()) !== null && s._receiver.write(e), s._receiver.end(), this[g] = void 0, clearTimeout(s._closeTimer), s._receiver._writableState.finished || s._receiver._writableState.errorEmitted ? s.emitClose() : (s._receiver.on("error", Fe), s._receiver.on("finish", Fe));
}
function te(s) {
  this[g]._receiver.write(s) || this.pause();
}
function Ze() {
  const s = this[g];
  s._readyState = d.CLOSING, s._receiver.end(), this.end();
}
function Qe() {
  const s = this[g];
  this.removeListener("error", Qe), this.on("error", Ye), s && (s._readyState = d.CLOSING, this.destroy());
}
const Ss = /* @__PURE__ */ et(ns), { tokenChars: Es } = J, { createHash: vs } = E;
export {
  Ss as WebSocket,
  Ss as default
};
