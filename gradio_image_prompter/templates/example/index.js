const { setContext: ee, getContext: p } = window.__gradio__svelte__internal, k = "WORKER_PROXY_CONTEXT_KEY";
function w() {
  return p(k);
}
function v(l) {
  return l.host === window.location.host || l.host === "localhost:7860" || l.host === "127.0.0.1:7860" || // Ref: https://github.com/gradio-app/gradio/blob/v3.32.0/js/app/src/Index.svelte#L194
  l.host === "lite.local";
}
async function f(l) {
  if (l == null)
    return l;
  const e = new URL(l);
  if (!v(e) || e.protocol !== "http:" && e.protocol !== "https:")
    return l;
  const o = w();
  if (o == null)
    return l;
  const n = e.pathname;
  return o.httpRequest({
    method: "GET",
    path: n,
    headers: {},
    query_string: ""
  }).then((t) => {
    if (t.status !== 200)
      throw new Error(`Failed to get file ${n} from the Wasm worker.`);
    const r = new Blob([t.body], {
      type: t.headers["Content-Type"]
    });
    return URL.createObjectURL(r);
  });
}
const {
  SvelteComponent: y,
  append: C,
  assign: _,
  compute_rest_props: d,
  detach: u,
  element: b,
  empty: R,
  exclude_internal_props: E,
  get_spread_update: O,
  handle_promise: h,
  init: q,
  insert: m,
  noop: c,
  safe_not_equal: T,
  set_attributes: g,
  set_data: P,
  set_style: U,
  src_url_equal: W,
  text: K,
  update_await_block_branch: X
} = window.__gradio__svelte__internal;
function Y(l) {
  let e, o = (
    /*error*/
    l[3].message + ""
  ), n;
  return {
    c() {
      e = b("p"), n = K(o), U(e, "color", "red");
    },
    m(t, r) {
      m(t, e, r), C(e, n);
    },
    p(t, r) {
      r & /*src*/
      1 && o !== (o = /*error*/
      t[3].message + "") && P(n, o);
    },
    d(t) {
      t && u(e);
    }
  };
}
function L(l) {
  let e, o, n = [
    {
      src: o = /*resolved_src*/
      l[2]
    },
    /*$$restProps*/
    l[1]
  ], t = {};
  for (let r = 0; r < n.length; r += 1)
    t = _(t, n[r]);
  return {
    c() {
      e = b("img"), g(e, t);
    },
    m(r, a) {
      m(r, e, a);
    },
    p(r, a) {
      g(e, t = O(n, [
        a & /*src*/
        1 && !W(e.src, o = /*resolved_src*/
        r[2]) && { src: o },
        a & /*$$restProps*/
        2 && /*$$restProps*/
        r[1]
      ]));
    },
    d(r) {
      r && u(e);
    }
  };
}
function N(l) {
  return { c, m: c, p: c, d: c };
}
function S(l) {
  let e, o, n = {
    ctx: l,
    current: null,
    token: null,
    hasCatch: !0,
    pending: N,
    then: L,
    catch: Y,
    value: 2,
    error: 3
  };
  return h(o = f(
    /*src*/
    l[0]
  ), n), {
    c() {
      e = R(), n.block.c();
    },
    m(t, r) {
      m(t, e, r), n.block.m(t, n.anchor = r), n.mount = () => e.parentNode, n.anchor = e;
    },
    p(t, [r]) {
      l = t, n.ctx = l, r & /*src*/
      1 && o !== (o = f(
        /*src*/
        l[0]
      )) && h(o, n) || X(n, l, r);
    },
    i: c,
    o: c,
    d(t) {
      t && u(e), n.block.d(t), n.token = null, n = null;
    }
  };
}
function j(l, e, o) {
  const n = ["src"];
  let t = d(e, n), { src: r = void 0 } = e;
  return l.$$set = (a) => {
    e = _(_({}, e), E(a)), o(1, t = d(e, n)), "src" in a && o(0, r = a.src);
  }, [r, t];
}
class B extends y {
  constructor(e) {
    super(), q(this, e, j, S, T, { src: 0 });
  }
}
const {
  SvelteComponent: F,
  attr: G,
  create_component: I,
  destroy_component: z,
  detach: A,
  element: D,
  init: H,
  insert: J,
  mount_component: M,
  safe_not_equal: Q,
  toggle_class: i,
  transition_in: V,
  transition_out: Z
} = window.__gradio__svelte__internal;
function x(l) {
  let e, o, n;
  return o = new B({
    props: {
      src: (
        /*samples_dir*/
        l[1] + /*value*/
        l[0]
      ),
      alt: ""
    }
  }), {
    c() {
      e = D("div"), I(o.$$.fragment), G(e, "class", "container svelte-h11ksk"), i(
        e,
        "table",
        /*type*/
        l[2] === "table"
      ), i(
        e,
        "gallery",
        /*type*/
        l[2] === "gallery"
      ), i(
        e,
        "selected",
        /*selected*/
        l[3]
      );
    },
    m(t, r) {
      J(t, e, r), M(o, e, null), n = !0;
    },
    p(t, [r]) {
      const a = {};
      r & /*samples_dir, value*/
      3 && (a.src = /*samples_dir*/
      t[1] + /*value*/
      t[0]), o.$set(a), (!n || r & /*type*/
      4) && i(
        e,
        "table",
        /*type*/
        t[2] === "table"
      ), (!n || r & /*type*/
      4) && i(
        e,
        "gallery",
        /*type*/
        t[2] === "gallery"
      ), (!n || r & /*selected*/
      8) && i(
        e,
        "selected",
        /*selected*/
        t[3]
      );
    },
    i(t) {
      n || (V(o.$$.fragment, t), n = !0);
    },
    o(t) {
      Z(o.$$.fragment, t), n = !1;
    },
    d(t) {
      t && A(e), z(o);
    }
  };
}
function $(l, e, o) {
  let { value: n } = e, { samples_dir: t } = e, { type: r } = e, { selected: a = !1 } = e;
  return l.$$set = (s) => {
    "value" in s && o(0, n = s.value), "samples_dir" in s && o(1, t = s.samples_dir), "type" in s && o(2, r = s.type), "selected" in s && o(3, a = s.selected);
  }, [n, t, r, a];
}
class te extends F {
  constructor(e) {
    super(), H(this, e, $, x, Q, {
      value: 0,
      samples_dir: 1,
      type: 2,
      selected: 3
    });
  }
}
export {
  te as default
};
