const Wl = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], Hn = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
Wl.reduce(
  (n, { color: e, primary: t, secondary: o }) => ({
    ...n,
    [e]: {
      primary: Hn[e][t],
      secondary: Hn[e][o]
    }
  }),
  {}
);
class At extends Error {
  constructor(e) {
    super(e), this.name = "ShareError";
  }
}
async function Vl(n, e) {
  if (window.__gradio_space__ == null)
    throw new At("Must be on Spaces to share.");
  let t, o, l;
  t = Gl(n), o = n.split(";")[0].split(":")[1], l = "file" + o.split("/")[1];
  const i = new File([t], l, { type: o }), s = await fetch("https://huggingface.co/uploads", {
    method: "POST",
    body: i,
    headers: {
      "Content-Type": i.type,
      "X-Requested-With": "XMLHttpRequest"
    }
  });
  if (!s.ok) {
    if (s.headers.get("content-type")?.includes("application/json")) {
      const a = await s.json();
      throw new At(`Upload failed: ${a.error}`);
    }
    throw new At("Upload failed.");
  }
  return await s.text();
}
function Gl(n) {
  for (var e = n.split(","), t = e[0].match(/:(.*?);/)[1], o = atob(e[1]), l = o.length, i = new Uint8Array(l); l--; )
    i[l] = o.charCodeAt(l);
  return new Blob([i], { type: t });
}
const {
  SvelteComponent: Hl,
  assign: Jl,
  create_slot: Xl,
  detach: Yl,
  element: Kl,
  get_all_dirty_from_scope: Ql,
  get_slot_changes: xl,
  get_spread_update: ei,
  init: ti,
  insert: ni,
  safe_not_equal: oi,
  set_dynamic_element_data: Jn,
  set_style: ue,
  toggle_class: Ae,
  transition_in: cl,
  transition_out: dl,
  update_slot_base: li
} = window.__gradio__svelte__internal;
function ii(n) {
  let e, t, o;
  const l = (
    /*#slots*/
    n[17].default
  ), i = Xl(
    l,
    n,
    /*$$scope*/
    n[16],
    null
  );
  let s = [
    { "data-testid": (
      /*test_id*/
      n[7]
    ) },
    { id: (
      /*elem_id*/
      n[2]
    ) },
    {
      class: t = "block " + /*elem_classes*/
      n[3].join(" ") + " svelte-1t38q2d"
    }
  ], r = {};
  for (let a = 0; a < s.length; a += 1)
    r = Jl(r, s[a]);
  return {
    c() {
      e = Kl(
        /*tag*/
        n[14]
      ), i && i.c(), Jn(
        /*tag*/
        n[14]
      )(e, r), Ae(
        e,
        "hidden",
        /*visible*/
        n[10] === !1
      ), Ae(
        e,
        "padded",
        /*padding*/
        n[6]
      ), Ae(
        e,
        "border_focus",
        /*border_mode*/
        n[5] === "focus"
      ), Ae(e, "hide-container", !/*explicit_call*/
      n[8] && !/*container*/
      n[9]), ue(e, "height", typeof /*height*/
      n[0] == "number" ? (
        /*height*/
        n[0] + "px"
      ) : void 0), ue(e, "width", typeof /*width*/
      n[1] == "number" ? `calc(min(${/*width*/
      n[1]}px, 100%))` : void 0), ue(
        e,
        "border-style",
        /*variant*/
        n[4]
      ), ue(
        e,
        "overflow",
        /*allow_overflow*/
        n[11] ? "visible" : "hidden"
      ), ue(
        e,
        "flex-grow",
        /*scale*/
        n[12]
      ), ue(e, "min-width", `calc(min(${/*min_width*/
      n[13]}px, 100%))`), ue(e, "border-width", "var(--block-border-width)");
    },
    m(a, _) {
      ni(a, e, _), i && i.m(e, null), o = !0;
    },
    p(a, _) {
      i && i.p && (!o || _ & /*$$scope*/
      65536) && li(
        i,
        l,
        a,
        /*$$scope*/
        a[16],
        o ? xl(
          l,
          /*$$scope*/
          a[16],
          _,
          null
        ) : Ql(
          /*$$scope*/
          a[16]
        ),
        null
      ), Jn(
        /*tag*/
        a[14]
      )(e, r = ei(s, [
        (!o || _ & /*test_id*/
        128) && { "data-testid": (
          /*test_id*/
          a[7]
        ) },
        (!o || _ & /*elem_id*/
        4) && { id: (
          /*elem_id*/
          a[2]
        ) },
        (!o || _ & /*elem_classes*/
        8 && t !== (t = "block " + /*elem_classes*/
        a[3].join(" ") + " svelte-1t38q2d")) && { class: t }
      ])), Ae(
        e,
        "hidden",
        /*visible*/
        a[10] === !1
      ), Ae(
        e,
        "padded",
        /*padding*/
        a[6]
      ), Ae(
        e,
        "border_focus",
        /*border_mode*/
        a[5] === "focus"
      ), Ae(e, "hide-container", !/*explicit_call*/
      a[8] && !/*container*/
      a[9]), _ & /*height*/
      1 && ue(e, "height", typeof /*height*/
      a[0] == "number" ? (
        /*height*/
        a[0] + "px"
      ) : void 0), _ & /*width*/
      2 && ue(e, "width", typeof /*width*/
      a[1] == "number" ? `calc(min(${/*width*/
      a[1]}px, 100%))` : void 0), _ & /*variant*/
      16 && ue(
        e,
        "border-style",
        /*variant*/
        a[4]
      ), _ & /*allow_overflow*/
      2048 && ue(
        e,
        "overflow",
        /*allow_overflow*/
        a[11] ? "visible" : "hidden"
      ), _ & /*scale*/
      4096 && ue(
        e,
        "flex-grow",
        /*scale*/
        a[12]
      ), _ & /*min_width*/
      8192 && ue(e, "min-width", `calc(min(${/*min_width*/
      a[13]}px, 100%))`);
    },
    i(a) {
      o || (cl(i, a), o = !0);
    },
    o(a) {
      dl(i, a), o = !1;
    },
    d(a) {
      a && Yl(e), i && i.d(a);
    }
  };
}
function si(n) {
  let e, t = (
    /*tag*/
    n[14] && ii(n)
  );
  return {
    c() {
      t && t.c();
    },
    m(o, l) {
      t && t.m(o, l), e = !0;
    },
    p(o, [l]) {
      /*tag*/
      o[14] && t.p(o, l);
    },
    i(o) {
      e || (cl(t, o), e = !0);
    },
    o(o) {
      dl(t, o), e = !1;
    },
    d(o) {
      t && t.d(o);
    }
  };
}
function ai(n, e, t) {
  let { $$slots: o = {}, $$scope: l } = e, { height: i = void 0 } = e, { width: s = void 0 } = e, { elem_id: r = "" } = e, { elem_classes: a = [] } = e, { variant: _ = "solid" } = e, { border_mode: u = "base" } = e, { padding: c = !0 } = e, { type: d = "normal" } = e, { test_id: f = void 0 } = e, { explicit_call: p = !1 } = e, { container: g = !0 } = e, { visible: S = !0 } = e, { allow_overflow: b = !0 } = e, { scale: y = null } = e, { min_width: h = 0 } = e, w = d === "fieldset" ? "fieldset" : "div";
  return n.$$set = (C) => {
    "height" in C && t(0, i = C.height), "width" in C && t(1, s = C.width), "elem_id" in C && t(2, r = C.elem_id), "elem_classes" in C && t(3, a = C.elem_classes), "variant" in C && t(4, _ = C.variant), "border_mode" in C && t(5, u = C.border_mode), "padding" in C && t(6, c = C.padding), "type" in C && t(15, d = C.type), "test_id" in C && t(7, f = C.test_id), "explicit_call" in C && t(8, p = C.explicit_call), "container" in C && t(9, g = C.container), "visible" in C && t(10, S = C.visible), "allow_overflow" in C && t(11, b = C.allow_overflow), "scale" in C && t(12, y = C.scale), "min_width" in C && t(13, h = C.min_width), "$$scope" in C && t(16, l = C.$$scope);
  }, [
    i,
    s,
    r,
    a,
    _,
    u,
    c,
    f,
    p,
    g,
    S,
    b,
    y,
    h,
    w,
    d,
    l,
    o
  ];
}
class ml extends Hl {
  constructor(e) {
    super(), ti(this, e, ai, si, oi, {
      height: 0,
      width: 1,
      elem_id: 2,
      elem_classes: 3,
      variant: 4,
      border_mode: 5,
      padding: 6,
      type: 15,
      test_id: 7,
      explicit_call: 8,
      container: 9,
      visible: 10,
      allow_overflow: 11,
      scale: 12,
      min_width: 13
    });
  }
}
const {
  SvelteComponent: qf,
  attr: Sf,
  create_slot: Cf,
  detach: Ef,
  element: Df,
  get_all_dirty_from_scope: Mf,
  get_slot_changes: zf,
  init: Nf,
  insert: If,
  safe_not_equal: Bf,
  transition_in: Tf,
  transition_out: Lf,
  update_slot_base: jf
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ff,
  attr: Pf,
  check_outros: Of,
  create_component: Af,
  create_slot: Rf,
  destroy_component: Uf,
  detach: Zf,
  element: Wf,
  empty: Vf,
  get_all_dirty_from_scope: Gf,
  get_slot_changes: Hf,
  group_outros: Jf,
  init: Xf,
  insert: Yf,
  mount_component: Kf,
  safe_not_equal: Qf,
  set_data: xf,
  space: ec,
  text: tc,
  toggle_class: nc,
  transition_in: oc,
  transition_out: lc,
  update_slot_base: ic
} = window.__gradio__svelte__internal, {
  SvelteComponent: ri,
  append: xt,
  attr: zt,
  create_component: _i,
  destroy_component: ui,
  detach: fi,
  element: Xn,
  init: ci,
  insert: di,
  mount_component: mi,
  safe_not_equal: pi,
  set_data: hi,
  space: gi,
  text: bi,
  toggle_class: Re,
  transition_in: wi,
  transition_out: vi
} = window.__gradio__svelte__internal;
function $i(n) {
  let e, t, o, l, i, s;
  return o = new /*Icon*/
  n[1]({}), {
    c() {
      e = Xn("label"), t = Xn("span"), _i(o.$$.fragment), l = gi(), i = bi(
        /*label*/
        n[0]
      ), zt(t, "class", "svelte-9gxdi0"), zt(e, "for", ""), zt(e, "data-testid", "block-label"), zt(e, "class", "svelte-9gxdi0"), Re(e, "hide", !/*show_label*/
      n[2]), Re(e, "sr-only", !/*show_label*/
      n[2]), Re(
        e,
        "float",
        /*float*/
        n[4]
      ), Re(
        e,
        "hide-label",
        /*disable*/
        n[3]
      );
    },
    m(r, a) {
      di(r, e, a), xt(e, t), mi(o, t, null), xt(e, l), xt(e, i), s = !0;
    },
    p(r, [a]) {
      (!s || a & /*label*/
      1) && hi(
        i,
        /*label*/
        r[0]
      ), (!s || a & /*show_label*/
      4) && Re(e, "hide", !/*show_label*/
      r[2]), (!s || a & /*show_label*/
      4) && Re(e, "sr-only", !/*show_label*/
      r[2]), (!s || a & /*float*/
      16) && Re(
        e,
        "float",
        /*float*/
        r[4]
      ), (!s || a & /*disable*/
      8) && Re(
        e,
        "hide-label",
        /*disable*/
        r[3]
      );
    },
    i(r) {
      s || (wi(o.$$.fragment, r), s = !0);
    },
    o(r) {
      vi(o.$$.fragment, r), s = !1;
    },
    d(r) {
      r && fi(e), ui(o);
    }
  };
}
function ki(n, e, t) {
  let { label: o = null } = e, { Icon: l } = e, { show_label: i = !0 } = e, { disable: s = !1 } = e, { float: r = !0 } = e;
  return n.$$set = (a) => {
    "label" in a && t(0, o = a.label), "Icon" in a && t(1, l = a.Icon), "show_label" in a && t(2, i = a.show_label), "disable" in a && t(3, s = a.disable), "float" in a && t(4, r = a.float);
  }, [o, l, i, s, r];
}
class pl extends ri {
  constructor(e) {
    super(), ci(this, e, ki, $i, pi, {
      label: 0,
      Icon: 1,
      show_label: 2,
      disable: 3,
      float: 4
    });
  }
}
const {
  SvelteComponent: yi,
  append: Mn,
  attr: Be,
  bubble: qi,
  create_component: Si,
  destroy_component: Ci,
  detach: hl,
  element: zn,
  init: Ei,
  insert: gl,
  listen: Di,
  mount_component: Mi,
  safe_not_equal: zi,
  set_data: Ni,
  set_style: Nt,
  space: Ii,
  text: Bi,
  toggle_class: he,
  transition_in: Ti,
  transition_out: Li
} = window.__gradio__svelte__internal;
function Yn(n) {
  let e, t;
  return {
    c() {
      e = zn("span"), t = Bi(
        /*label*/
        n[1]
      ), Be(e, "class", "svelte-lpi64a");
    },
    m(o, l) {
      gl(o, e, l), Mn(e, t);
    },
    p(o, l) {
      l & /*label*/
      2 && Ni(
        t,
        /*label*/
        o[1]
      );
    },
    d(o) {
      o && hl(e);
    }
  };
}
function ji(n) {
  let e, t, o, l, i, s, r, a = (
    /*show_label*/
    n[2] && Yn(n)
  );
  return l = new /*Icon*/
  n[0]({}), {
    c() {
      e = zn("button"), a && a.c(), t = Ii(), o = zn("div"), Si(l.$$.fragment), Be(o, "class", "svelte-lpi64a"), he(
        o,
        "small",
        /*size*/
        n[4] === "small"
      ), he(
        o,
        "large",
        /*size*/
        n[4] === "large"
      ), e.disabled = /*disabled*/
      n[7], Be(
        e,
        "aria-label",
        /*label*/
        n[1]
      ), Be(
        e,
        "aria-haspopup",
        /*hasPopup*/
        n[8]
      ), Be(
        e,
        "title",
        /*label*/
        n[1]
      ), Be(e, "class", "svelte-lpi64a"), he(
        e,
        "pending",
        /*pending*/
        n[3]
      ), he(
        e,
        "padded",
        /*padded*/
        n[5]
      ), he(
        e,
        "highlight",
        /*highlight*/
        n[6]
      ), he(
        e,
        "transparent",
        /*transparent*/
        n[9]
      ), Nt(e, "color", !/*disabled*/
      n[7] && /*_color*/
      n[11] ? (
        /*_color*/
        n[11]
      ) : "var(--block-label-text-color)"), Nt(e, "--bg-color", /*disabled*/
      n[7] ? "auto" : (
        /*background*/
        n[10]
      ));
    },
    m(_, u) {
      gl(_, e, u), a && a.m(e, null), Mn(e, t), Mn(e, o), Mi(l, o, null), i = !0, s || (r = Di(
        e,
        "click",
        /*click_handler*/
        n[13]
      ), s = !0);
    },
    p(_, [u]) {
      /*show_label*/
      _[2] ? a ? a.p(_, u) : (a = Yn(_), a.c(), a.m(e, t)) : a && (a.d(1), a = null), (!i || u & /*size*/
      16) && he(
        o,
        "small",
        /*size*/
        _[4] === "small"
      ), (!i || u & /*size*/
      16) && he(
        o,
        "large",
        /*size*/
        _[4] === "large"
      ), (!i || u & /*disabled*/
      128) && (e.disabled = /*disabled*/
      _[7]), (!i || u & /*label*/
      2) && Be(
        e,
        "aria-label",
        /*label*/
        _[1]
      ), (!i || u & /*hasPopup*/
      256) && Be(
        e,
        "aria-haspopup",
        /*hasPopup*/
        _[8]
      ), (!i || u & /*label*/
      2) && Be(
        e,
        "title",
        /*label*/
        _[1]
      ), (!i || u & /*pending*/
      8) && he(
        e,
        "pending",
        /*pending*/
        _[3]
      ), (!i || u & /*padded*/
      32) && he(
        e,
        "padded",
        /*padded*/
        _[5]
      ), (!i || u & /*highlight*/
      64) && he(
        e,
        "highlight",
        /*highlight*/
        _[6]
      ), (!i || u & /*transparent*/
      512) && he(
        e,
        "transparent",
        /*transparent*/
        _[9]
      ), u & /*disabled, _color*/
      2176 && Nt(e, "color", !/*disabled*/
      _[7] && /*_color*/
      _[11] ? (
        /*_color*/
        _[11]
      ) : "var(--block-label-text-color)"), u & /*disabled, background*/
      1152 && Nt(e, "--bg-color", /*disabled*/
      _[7] ? "auto" : (
        /*background*/
        _[10]
      ));
    },
    i(_) {
      i || (Ti(l.$$.fragment, _), i = !0);
    },
    o(_) {
      Li(l.$$.fragment, _), i = !1;
    },
    d(_) {
      _ && hl(e), a && a.d(), Ci(l), s = !1, r();
    }
  };
}
function Fi(n, e, t) {
  let o, { Icon: l } = e, { label: i = "" } = e, { show_label: s = !1 } = e, { pending: r = !1 } = e, { size: a = "small" } = e, { padded: _ = !0 } = e, { highlight: u = !1 } = e, { disabled: c = !1 } = e, { hasPopup: d = !1 } = e, { color: f = "var(--block-label-text-color)" } = e, { transparent: p = !1 } = e, { background: g = "var(--background-fill-primary)" } = e;
  function S(b) {
    qi.call(this, n, b);
  }
  return n.$$set = (b) => {
    "Icon" in b && t(0, l = b.Icon), "label" in b && t(1, i = b.label), "show_label" in b && t(2, s = b.show_label), "pending" in b && t(3, r = b.pending), "size" in b && t(4, a = b.size), "padded" in b && t(5, _ = b.padded), "highlight" in b && t(6, u = b.highlight), "disabled" in b && t(7, c = b.disabled), "hasPopup" in b && t(8, d = b.hasPopup), "color" in b && t(12, f = b.color), "transparent" in b && t(9, p = b.transparent), "background" in b && t(10, g = b.background);
  }, n.$$.update = () => {
    n.$$.dirty & /*highlight, color*/
    4160 && t(11, o = u ? "var(--color-accent)" : f);
  }, [
    l,
    i,
    s,
    r,
    a,
    _,
    u,
    c,
    d,
    p,
    g,
    o,
    f,
    S
  ];
}
class at extends yi {
  constructor(e) {
    super(), Ei(this, e, Fi, ji, zi, {
      Icon: 0,
      label: 1,
      show_label: 2,
      pending: 3,
      size: 4,
      padded: 5,
      highlight: 6,
      disabled: 7,
      hasPopup: 8,
      color: 12,
      transparent: 9,
      background: 10
    });
  }
}
const {
  SvelteComponent: Pi,
  append: Oi,
  attr: en,
  binding_callbacks: Ai,
  create_slot: Ri,
  detach: Ui,
  element: Kn,
  get_all_dirty_from_scope: Zi,
  get_slot_changes: Wi,
  init: Vi,
  insert: Gi,
  safe_not_equal: Hi,
  toggle_class: Ue,
  transition_in: Ji,
  transition_out: Xi,
  update_slot_base: Yi
} = window.__gradio__svelte__internal;
function Ki(n) {
  let e, t, o;
  const l = (
    /*#slots*/
    n[5].default
  ), i = Ri(
    l,
    n,
    /*$$scope*/
    n[4],
    null
  );
  return {
    c() {
      e = Kn("div"), t = Kn("div"), i && i.c(), en(t, "class", "icon svelte-3w3rth"), en(e, "class", "empty svelte-3w3rth"), en(e, "aria-label", "Empty value"), Ue(
        e,
        "small",
        /*size*/
        n[0] === "small"
      ), Ue(
        e,
        "large",
        /*size*/
        n[0] === "large"
      ), Ue(
        e,
        "unpadded_box",
        /*unpadded_box*/
        n[1]
      ), Ue(
        e,
        "small_parent",
        /*parent_height*/
        n[3]
      );
    },
    m(s, r) {
      Gi(s, e, r), Oi(e, t), i && i.m(t, null), n[6](e), o = !0;
    },
    p(s, [r]) {
      i && i.p && (!o || r & /*$$scope*/
      16) && Yi(
        i,
        l,
        s,
        /*$$scope*/
        s[4],
        o ? Wi(
          l,
          /*$$scope*/
          s[4],
          r,
          null
        ) : Zi(
          /*$$scope*/
          s[4]
        ),
        null
      ), (!o || r & /*size*/
      1) && Ue(
        e,
        "small",
        /*size*/
        s[0] === "small"
      ), (!o || r & /*size*/
      1) && Ue(
        e,
        "large",
        /*size*/
        s[0] === "large"
      ), (!o || r & /*unpadded_box*/
      2) && Ue(
        e,
        "unpadded_box",
        /*unpadded_box*/
        s[1]
      ), (!o || r & /*parent_height*/
      8) && Ue(
        e,
        "small_parent",
        /*parent_height*/
        s[3]
      );
    },
    i(s) {
      o || (Ji(i, s), o = !0);
    },
    o(s) {
      Xi(i, s), o = !1;
    },
    d(s) {
      s && Ui(e), i && i.d(s), n[6](null);
    }
  };
}
function Qi(n, e, t) {
  let o, { $$slots: l = {}, $$scope: i } = e, { size: s = "small" } = e, { unpadded_box: r = !1 } = e, a;
  function _(c) {
    var d;
    if (!c) return !1;
    const { height: f } = c.getBoundingClientRect(), { height: p } = ((d = c.parentElement) === null || d === void 0 ? void 0 : d.getBoundingClientRect()) || { height: f };
    return f > p + 2;
  }
  function u(c) {
    Ai[c ? "unshift" : "push"](() => {
      a = c, t(2, a);
    });
  }
  return n.$$set = (c) => {
    "size" in c && t(0, s = c.size), "unpadded_box" in c && t(1, r = c.unpadded_box), "$$scope" in c && t(4, i = c.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty & /*el*/
    4 && t(3, o = _(a));
  }, [s, r, a, o, i, l, u];
}
class bl extends Pi {
  constructor(e) {
    super(), Vi(this, e, Qi, Ki, Hi, { size: 0, unpadded_box: 1 });
  }
}
const {
  SvelteComponent: sc,
  append: ac,
  attr: rc,
  detach: _c,
  init: uc,
  insert: fc,
  noop: cc,
  safe_not_equal: dc,
  svg_element: mc
} = window.__gradio__svelte__internal, {
  SvelteComponent: pc,
  append: hc,
  attr: gc,
  detach: bc,
  init: wc,
  insert: vc,
  noop: $c,
  safe_not_equal: kc,
  svg_element: yc
} = window.__gradio__svelte__internal, {
  SvelteComponent: qc,
  append: Sc,
  attr: Cc,
  detach: Ec,
  init: Dc,
  insert: Mc,
  noop: zc,
  safe_not_equal: Nc,
  svg_element: Ic
} = window.__gradio__svelte__internal, {
  SvelteComponent: Bc,
  append: Tc,
  attr: Lc,
  detach: jc,
  init: Fc,
  insert: Pc,
  noop: Oc,
  safe_not_equal: Ac,
  svg_element: Rc
} = window.__gradio__svelte__internal, {
  SvelteComponent: Uc,
  append: Zc,
  attr: Wc,
  detach: Vc,
  init: Gc,
  insert: Hc,
  noop: Jc,
  safe_not_equal: Xc,
  svg_element: Yc
} = window.__gradio__svelte__internal, {
  SvelteComponent: Kc,
  append: Qc,
  attr: xc,
  detach: ed,
  init: td,
  insert: nd,
  noop: od,
  safe_not_equal: ld,
  svg_element: id
} = window.__gradio__svelte__internal, {
  SvelteComponent: sd,
  append: ad,
  attr: rd,
  detach: _d,
  init: ud,
  insert: fd,
  noop: cd,
  safe_not_equal: dd,
  svg_element: md
} = window.__gradio__svelte__internal, {
  SvelteComponent: pd,
  append: hd,
  attr: gd,
  detach: bd,
  init: wd,
  insert: vd,
  noop: $d,
  safe_not_equal: kd,
  svg_element: yd
} = window.__gradio__svelte__internal, {
  SvelteComponent: xi,
  append: tn,
  attr: qe,
  detach: es,
  init: ts,
  insert: ns,
  noop: nn,
  safe_not_equal: os,
  set_style: ze,
  svg_element: It
} = window.__gradio__svelte__internal;
function ls(n) {
  let e, t, o, l;
  return {
    c() {
      e = It("svg"), t = It("g"), o = It("path"), l = It("path"), qe(o, "d", "M18,6L6.087,17.913"), ze(o, "fill", "none"), ze(o, "fill-rule", "nonzero"), ze(o, "stroke-width", "2px"), qe(t, "transform", "matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"), qe(l, "d", "M4.364,4.364L19.636,19.636"), ze(l, "fill", "none"), ze(l, "fill-rule", "nonzero"), ze(l, "stroke-width", "2px"), qe(e, "width", "100%"), qe(e, "height", "100%"), qe(e, "viewBox", "0 0 24 24"), qe(e, "version", "1.1"), qe(e, "xmlns", "http://www.w3.org/2000/svg"), qe(e, "xmlns:xlink", "http://www.w3.org/1999/xlink"), qe(e, "xml:space", "preserve"), qe(e, "stroke", "currentColor"), ze(e, "fill-rule", "evenodd"), ze(e, "clip-rule", "evenodd"), ze(e, "stroke-linecap", "round"), ze(e, "stroke-linejoin", "round");
    },
    m(i, s) {
      ns(i, e, s), tn(e, t), tn(t, o), tn(e, l);
    },
    p: nn,
    i: nn,
    o: nn,
    d(i) {
      i && es(e);
    }
  };
}
class is extends xi {
  constructor(e) {
    super(), ts(this, e, null, ls, os, {});
  }
}
const {
  SvelteComponent: qd,
  append: Sd,
  attr: Cd,
  detach: Ed,
  init: Dd,
  insert: Md,
  noop: zd,
  safe_not_equal: Nd,
  svg_element: Id
} = window.__gradio__svelte__internal, {
  SvelteComponent: Bd,
  append: Td,
  attr: Ld,
  detach: jd,
  init: Fd,
  insert: Pd,
  noop: Od,
  safe_not_equal: Ad,
  svg_element: Rd
} = window.__gradio__svelte__internal, {
  SvelteComponent: ss,
  append: as,
  attr: vt,
  detach: rs,
  init: _s,
  insert: us,
  noop: on,
  safe_not_equal: fs,
  svg_element: Qn
} = window.__gradio__svelte__internal;
function cs(n) {
  let e, t;
  return {
    c() {
      e = Qn("svg"), t = Qn("path"), vt(t, "d", "M23,20a5,5,0,0,0-3.89,1.89L11.8,17.32a4.46,4.46,0,0,0,0-2.64l7.31-4.57A5,5,0,1,0,18,7a4.79,4.79,0,0,0,.2,1.32l-7.31,4.57a5,5,0,1,0,0,6.22l7.31,4.57A4.79,4.79,0,0,0,18,25a5,5,0,1,0,5-5ZM23,4a3,3,0,1,1-3,3A3,3,0,0,1,23,4ZM7,19a3,3,0,1,1,3-3A3,3,0,0,1,7,19Zm16,9a3,3,0,1,1,3-3A3,3,0,0,1,23,28Z"), vt(t, "fill", "currentColor"), vt(e, "id", "icon"), vt(e, "xmlns", "http://www.w3.org/2000/svg"), vt(e, "viewBox", "0 0 32 32");
    },
    m(o, l) {
      us(o, e, l), as(e, t);
    },
    p: on,
    i: on,
    o: on,
    d(o) {
      o && rs(e);
    }
  };
}
class ds extends ss {
  constructor(e) {
    super(), _s(this, e, null, cs, fs, {});
  }
}
const {
  SvelteComponent: Ud,
  append: Zd,
  attr: Wd,
  detach: Vd,
  init: Gd,
  insert: Hd,
  noop: Jd,
  safe_not_equal: Xd,
  svg_element: Yd
} = window.__gradio__svelte__internal, {
  SvelteComponent: Kd,
  append: Qd,
  attr: xd,
  detach: e0,
  init: t0,
  insert: n0,
  noop: o0,
  safe_not_equal: l0,
  svg_element: i0
} = window.__gradio__svelte__internal, {
  SvelteComponent: s0,
  append: a0,
  attr: r0,
  detach: _0,
  init: u0,
  insert: f0,
  noop: c0,
  safe_not_equal: d0,
  svg_element: m0
} = window.__gradio__svelte__internal, {
  SvelteComponent: ms,
  append: ps,
  attr: et,
  detach: hs,
  init: gs,
  insert: bs,
  noop: ln,
  safe_not_equal: ws,
  svg_element: xn
} = window.__gradio__svelte__internal;
function vs(n) {
  let e, t;
  return {
    c() {
      e = xn("svg"), t = xn("path"), et(t, "fill", "currentColor"), et(t, "d", "M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10l10-10z"), et(e, "xmlns", "http://www.w3.org/2000/svg"), et(e, "width", "100%"), et(e, "height", "100%"), et(e, "viewBox", "0 0 32 32");
    },
    m(o, l) {
      bs(o, e, l), ps(e, t);
    },
    p: ln,
    i: ln,
    o: ln,
    d(o) {
      o && hs(e);
    }
  };
}
class $s extends ms {
  constructor(e) {
    super(), gs(this, e, null, vs, ws, {});
  }
}
const {
  SvelteComponent: p0,
  append: h0,
  attr: g0,
  detach: b0,
  init: w0,
  insert: v0,
  noop: $0,
  safe_not_equal: k0,
  svg_element: y0
} = window.__gradio__svelte__internal, {
  SvelteComponent: q0,
  append: S0,
  attr: C0,
  detach: E0,
  init: D0,
  insert: M0,
  noop: z0,
  safe_not_equal: N0,
  svg_element: I0
} = window.__gradio__svelte__internal, {
  SvelteComponent: ks,
  append: sn,
  attr: Se,
  detach: ys,
  init: qs,
  insert: Ss,
  noop: an,
  safe_not_equal: Cs,
  svg_element: Bt
} = window.__gradio__svelte__internal;
function Es(n) {
  let e, t, o, l;
  return {
    c() {
      e = Bt("svg"), t = Bt("g"), o = Bt("path"), l = Bt("path"), Se(o, "fill", "currentColor"), Se(o, "d", "m5.505 11.41l.53.53l-.53-.53ZM3 14.952h-.75H3ZM9.048 21v.75V21ZM11.41 5.505l-.53-.53l.53.53Zm1.831 12.34a.75.75 0 0 0 1.06-1.061l-1.06 1.06ZM7.216 9.697a.75.75 0 1 0-1.06 1.061l1.06-1.06Zm10.749 2.362l-5.905 5.905l1.06 1.06l5.905-5.904l-1.06-1.06Zm-11.93-.12l5.905-5.905l-1.06-1.06l-5.905 5.904l1.06 1.06Zm0 6.025c-.85-.85-1.433-1.436-1.812-1.933c-.367-.481-.473-.79-.473-1.08h-1.5c0 .749.312 1.375.78 1.99c.455.596 1.125 1.263 1.945 2.083l1.06-1.06Zm-1.06-7.086c-.82.82-1.49 1.488-1.945 2.084c-.468.614-.78 1.24-.78 1.99h1.5c0-.29.106-.6.473-1.08c.38-.498.962-1.083 1.812-1.933l-1.06-1.06Zm7.085 7.086c-.85.85-1.435 1.433-1.933 1.813c-.48.366-.79.472-1.08.472v1.5c.75 0 1.376-.312 1.99-.78c.596-.455 1.264-1.125 2.084-1.945l-1.06-1.06Zm-7.085 1.06c.82.82 1.487 1.49 2.084 1.945c.614.468 1.24.78 1.989.78v-1.5c-.29 0-.599-.106-1.08-.473c-.497-.38-1.083-.962-1.933-1.812l-1.06 1.06Zm12.99-12.99c.85.85 1.433 1.436 1.813 1.933c.366.481.472.79.472 1.08h1.5c0-.749-.312-1.375-.78-1.99c-.455-.596-1.125-1.263-1.945-2.083l-1.06 1.06Zm1.06 7.086c.82-.82 1.49-1.488 1.945-2.084c.468-.614.78-1.24.78-1.99h-1.5c0 .29-.106.6-.473 1.08c-.38.498-.962 1.083-1.812 1.933l1.06 1.06Zm0-8.146c-.82-.82-1.487-1.49-2.084-1.945c-.614-.468-1.24-.78-1.989-.78v1.5c.29 0 .599.106 1.08.473c.497.38 1.083.962 1.933 1.812l1.06-1.06Zm-7.085 1.06c.85-.85 1.435-1.433 1.933-1.812c.48-.367.79-.473 1.08-.473v-1.5c-.75 0-1.376.312-1.99.78c-.596.455-1.264 1.125-2.084 1.945l1.06 1.06Zm2.362 10.749L7.216 9.698l-1.06 1.061l7.085 7.085l1.06-1.06Z"), Se(l, "stroke", "currentColor"), Se(l, "stroke-linecap", "round"), Se(l, "stroke-width", "1.5"), Se(l, "d", "M9 21h12"), Se(t, "fill", "none"), Se(e, "xmlns", "http://www.w3.org/2000/svg"), Se(e, "width", "100%"), Se(e, "height", "100%"), Se(e, "viewBox", "0 0 24 24");
    },
    m(i, s) {
      Ss(i, e, s), sn(e, t), sn(t, o), sn(t, l);
    },
    p: an,
    i: an,
    o: an,
    d(i) {
      i && ys(e);
    }
  };
}
class Ds extends ks {
  constructor(e) {
    super(), qs(this, e, null, Es, Cs, {});
  }
}
const {
  SvelteComponent: B0,
  append: T0,
  attr: L0,
  detach: j0,
  init: F0,
  insert: P0,
  noop: O0,
  safe_not_equal: A0,
  svg_element: R0
} = window.__gradio__svelte__internal, {
  SvelteComponent: U0,
  append: Z0,
  attr: W0,
  detach: V0,
  init: G0,
  insert: H0,
  noop: J0,
  safe_not_equal: X0,
  svg_element: Y0
} = window.__gradio__svelte__internal, {
  SvelteComponent: K0,
  append: Q0,
  attr: x0,
  detach: e1,
  init: t1,
  insert: n1,
  noop: o1,
  safe_not_equal: l1,
  svg_element: i1
} = window.__gradio__svelte__internal, {
  SvelteComponent: s1,
  append: a1,
  attr: r1,
  detach: _1,
  init: u1,
  insert: f1,
  noop: c1,
  safe_not_equal: d1,
  svg_element: m1
} = window.__gradio__svelte__internal, {
  SvelteComponent: p1,
  append: h1,
  attr: g1,
  detach: b1,
  init: w1,
  insert: v1,
  noop: $1,
  safe_not_equal: k1,
  svg_element: y1
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ms,
  append: rn,
  attr: x,
  detach: zs,
  init: Ns,
  insert: Is,
  noop: _n,
  safe_not_equal: Bs,
  svg_element: Tt
} = window.__gradio__svelte__internal;
function Ts(n) {
  let e, t, o, l;
  return {
    c() {
      e = Tt("svg"), t = Tt("rect"), o = Tt("circle"), l = Tt("polyline"), x(t, "x", "3"), x(t, "y", "3"), x(t, "width", "18"), x(t, "height", "18"), x(t, "rx", "2"), x(t, "ry", "2"), x(o, "cx", "8.5"), x(o, "cy", "8.5"), x(o, "r", "1.5"), x(l, "points", "21 15 16 10 5 21"), x(e, "xmlns", "http://www.w3.org/2000/svg"), x(e, "width", "100%"), x(e, "height", "100%"), x(e, "viewBox", "0 0 24 24"), x(e, "fill", "none"), x(e, "stroke", "currentColor"), x(e, "stroke-width", "1.5"), x(e, "stroke-linecap", "round"), x(e, "stroke-linejoin", "round"), x(e, "class", "feather feather-image");
    },
    m(i, s) {
      Is(i, e, s), rn(e, t), rn(e, o), rn(e, l);
    },
    p: _n,
    i: _n,
    o: _n,
    d(i) {
      i && zs(e);
    }
  };
}
let Yt = class extends Ms {
  constructor(e) {
    super(), Ns(this, e, null, Ts, Bs, {});
  }
};
const {
  SvelteComponent: Ls,
  append: js,
  attr: tt,
  detach: Fs,
  init: Ps,
  insert: Os,
  noop: un,
  safe_not_equal: As,
  svg_element: eo
} = window.__gradio__svelte__internal;
function Rs(n) {
  let e, t;
  return {
    c() {
      e = eo("svg"), t = eo("path"), tt(t, "fill", "currentColor"), tt(t, "d", "M13.75 2a2.25 2.25 0 0 1 2.236 2.002V4h1.764A2.25 2.25 0 0 1 20 6.25V11h-1.5V6.25a.75.75 0 0 0-.75-.75h-2.129c-.404.603-1.091 1-1.871 1h-3.5c-.78 0-1.467-.397-1.871-1H6.25a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h4.78a3.99 3.99 0 0 0 .505 1.5H6.25A2.25 2.25 0 0 1 4 19.75V6.25A2.25 2.25 0 0 1 6.25 4h1.764a2.25 2.25 0 0 1 2.236-2h3.5Zm2.245 2.096L16 4.25c0-.052-.002-.103-.005-.154ZM13.75 3.5h-3.5a.75.75 0 0 0 0 1.5h3.5a.75.75 0 0 0 0-1.5ZM15 12a3 3 0 0 0-3 3v5c0 .556.151 1.077.415 1.524l3.494-3.494a2.25 2.25 0 0 1 3.182 0l3.494 3.494c.264-.447.415-.968.415-1.524v-5a3 3 0 0 0-3-3h-5Zm0 11a2.985 2.985 0 0 1-1.524-.415l3.494-3.494a.75.75 0 0 1 1.06 0l3.494 3.494A2.985 2.985 0 0 1 20 23h-5Zm5-7a1 1 0 1 1 0-2a1 1 0 0 1 0 2Z"), tt(e, "xmlns", "http://www.w3.org/2000/svg"), tt(e, "width", "100%"), tt(e, "height", "100%"), tt(e, "viewBox", "0 0 24 24");
    },
    m(o, l) {
      Os(o, e, l), js(e, t);
    },
    p: un,
    i: un,
    o: un,
    d(o) {
      o && Fs(e);
    }
  };
}
class Us extends Ls {
  constructor(e) {
    super(), Ps(this, e, null, Rs, As, {});
  }
}
const {
  SvelteComponent: S1,
  append: C1,
  attr: E1,
  detach: D1,
  init: M1,
  insert: z1,
  noop: N1,
  safe_not_equal: I1,
  svg_element: B1
} = window.__gradio__svelte__internal, {
  SvelteComponent: T1,
  append: L1,
  attr: j1,
  detach: F1,
  init: P1,
  insert: O1,
  noop: A1,
  safe_not_equal: R1,
  svg_element: U1
} = window.__gradio__svelte__internal, {
  SvelteComponent: Z1,
  append: W1,
  attr: V1,
  detach: G1,
  init: H1,
  insert: J1,
  noop: X1,
  safe_not_equal: Y1,
  svg_element: K1
} = window.__gradio__svelte__internal, {
  SvelteComponent: Q1,
  append: x1,
  attr: em,
  detach: tm,
  init: nm,
  insert: om,
  noop: lm,
  safe_not_equal: im,
  svg_element: sm
} = window.__gradio__svelte__internal, {
  SvelteComponent: am,
  append: rm,
  attr: _m,
  detach: um,
  init: fm,
  insert: cm,
  noop: dm,
  safe_not_equal: mm,
  svg_element: pm
} = window.__gradio__svelte__internal, {
  SvelteComponent: hm,
  append: gm,
  attr: bm,
  detach: wm,
  init: vm,
  insert: $m,
  noop: km,
  safe_not_equal: ym,
  svg_element: qm
} = window.__gradio__svelte__internal, {
  SvelteComponent: Sm,
  append: Cm,
  attr: Em,
  detach: Dm,
  init: Mm,
  insert: zm,
  noop: Nm,
  safe_not_equal: Im,
  svg_element: Bm
} = window.__gradio__svelte__internal, {
  SvelteComponent: Tm,
  append: Lm,
  attr: jm,
  detach: Fm,
  init: Pm,
  insert: Om,
  noop: Am,
  safe_not_equal: Rm,
  svg_element: Um
} = window.__gradio__svelte__internal, {
  SvelteComponent: Zm,
  append: Wm,
  attr: Vm,
  detach: Gm,
  init: Hm,
  insert: Jm,
  noop: Xm,
  safe_not_equal: Ym,
  svg_element: Km
} = window.__gradio__svelte__internal, {
  SvelteComponent: Qm,
  append: xm,
  attr: ep,
  detach: tp,
  init: np,
  insert: op,
  noop: lp,
  safe_not_equal: ip,
  svg_element: sp
} = window.__gradio__svelte__internal, {
  SvelteComponent: ap,
  append: rp,
  attr: _p,
  detach: up,
  init: fp,
  insert: cp,
  noop: dp,
  safe_not_equal: mp,
  set_style: pp,
  svg_element: hp
} = window.__gradio__svelte__internal, {
  SvelteComponent: gp,
  append: bp,
  attr: wp,
  detach: vp,
  init: $p,
  insert: kp,
  noop: yp,
  safe_not_equal: qp,
  svg_element: Sp
} = window.__gradio__svelte__internal, {
  SvelteComponent: Cp,
  append: Ep,
  attr: Dp,
  detach: Mp,
  init: zp,
  insert: Np,
  noop: Ip,
  safe_not_equal: Bp,
  svg_element: Tp
} = window.__gradio__svelte__internal, {
  SvelteComponent: Lp,
  append: jp,
  attr: Fp,
  detach: Pp,
  init: Op,
  insert: Ap,
  noop: Rp,
  safe_not_equal: Up,
  svg_element: Zp
} = window.__gradio__svelte__internal, {
  SvelteComponent: Wp,
  append: Vp,
  attr: Gp,
  detach: Hp,
  init: Jp,
  insert: Xp,
  noop: Yp,
  safe_not_equal: Kp,
  svg_element: Qp
} = window.__gradio__svelte__internal, {
  SvelteComponent: xp,
  append: eh,
  attr: th,
  detach: nh,
  init: oh,
  insert: lh,
  noop: ih,
  safe_not_equal: sh,
  svg_element: ah
} = window.__gradio__svelte__internal, {
  SvelteComponent: rh,
  append: _h,
  attr: uh,
  detach: fh,
  init: ch,
  insert: dh,
  noop: mh,
  safe_not_equal: ph,
  svg_element: hh
} = window.__gradio__svelte__internal, {
  SvelteComponent: gh,
  append: bh,
  attr: wh,
  detach: vh,
  init: $h,
  insert: kh,
  noop: yh,
  safe_not_equal: qh,
  svg_element: Sh
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ch,
  append: Eh,
  attr: Dh,
  detach: Mh,
  init: zh,
  insert: Nh,
  noop: Ih,
  safe_not_equal: Bh,
  svg_element: Th,
  text: Lh
} = window.__gradio__svelte__internal, {
  SvelteComponent: jh,
  append: Fh,
  attr: Ph,
  detach: Oh,
  init: Ah,
  insert: Rh,
  noop: Uh,
  safe_not_equal: Zh,
  svg_element: Wh
} = window.__gradio__svelte__internal, {
  SvelteComponent: Vh,
  append: Gh,
  attr: Hh,
  detach: Jh,
  init: Xh,
  insert: Yh,
  noop: Kh,
  safe_not_equal: Qh,
  svg_element: xh
} = window.__gradio__svelte__internal, {
  SvelteComponent: Zs,
  append: to,
  attr: ge,
  detach: Ws,
  init: Vs,
  insert: Gs,
  noop: fn,
  safe_not_equal: Hs,
  svg_element: cn
} = window.__gradio__svelte__internal;
function Js(n) {
  let e, t, o;
  return {
    c() {
      e = cn("svg"), t = cn("polyline"), o = cn("path"), ge(t, "points", "1 4 1 10 7 10"), ge(o, "d", "M3.51 15a9 9 0 1 0 2.13-9.36L1 10"), ge(e, "xmlns", "http://www.w3.org/2000/svg"), ge(e, "width", "100%"), ge(e, "height", "100%"), ge(e, "viewBox", "0 0 24 24"), ge(e, "fill", "none"), ge(e, "stroke", "currentColor"), ge(e, "stroke-width", "2"), ge(e, "stroke-linecap", "round"), ge(e, "stroke-linejoin", "round"), ge(e, "class", "feather feather-rotate-ccw");
    },
    m(l, i) {
      Gs(l, e, i), to(e, t), to(e, o);
    },
    p: fn,
    i: fn,
    o: fn,
    d(l) {
      l && Ws(e);
    }
  };
}
class Xs extends Zs {
  constructor(e) {
    super(), Vs(this, e, null, Js, Hs, {});
  }
}
const {
  SvelteComponent: Ys,
  append: dn,
  attr: se,
  detach: Ks,
  init: Qs,
  insert: xs,
  noop: mn,
  safe_not_equal: ea,
  svg_element: Lt
} = window.__gradio__svelte__internal;
function ta(n) {
  let e, t, o, l;
  return {
    c() {
      e = Lt("svg"), t = Lt("path"), o = Lt("polyline"), l = Lt("line"), se(t, "d", "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"), se(o, "points", "17 8 12 3 7 8"), se(l, "x1", "12"), se(l, "y1", "3"), se(l, "x2", "12"), se(l, "y2", "15"), se(e, "xmlns", "http://www.w3.org/2000/svg"), se(e, "width", "90%"), se(e, "height", "90%"), se(e, "viewBox", "0 0 24 24"), se(e, "fill", "none"), se(e, "stroke", "currentColor"), se(e, "stroke-width", "2"), se(e, "stroke-linecap", "round"), se(e, "stroke-linejoin", "round"), se(e, "class", "feather feather-upload");
    },
    m(i, s) {
      xs(i, e, s), dn(e, t), dn(e, o), dn(e, l);
    },
    p: mn,
    i: mn,
    o: mn,
    d(i) {
      i && Ks(e);
    }
  };
}
let wl = class extends Ys {
  constructor(e) {
    super(), Qs(this, e, null, ta, ea, {});
  }
};
const {
  SvelteComponent: tg,
  append: ng,
  attr: og,
  detach: lg,
  init: ig,
  insert: sg,
  noop: ag,
  safe_not_equal: rg,
  svg_element: _g
} = window.__gradio__svelte__internal, {
  SvelteComponent: ug,
  append: fg,
  attr: cg,
  detach: dg,
  init: mg,
  insert: pg,
  noop: hg,
  safe_not_equal: gg,
  svg_element: bg,
  text: wg
} = window.__gradio__svelte__internal, {
  SvelteComponent: vg,
  append: $g,
  attr: kg,
  detach: yg,
  init: qg,
  insert: Sg,
  noop: Cg,
  safe_not_equal: Eg,
  svg_element: Dg,
  text: Mg
} = window.__gradio__svelte__internal, {
  SvelteComponent: zg,
  append: Ng,
  attr: Ig,
  detach: Bg,
  init: Tg,
  insert: Lg,
  noop: jg,
  safe_not_equal: Fg,
  svg_element: Pg,
  text: Og
} = window.__gradio__svelte__internal, {
  SvelteComponent: Ag,
  append: Rg,
  attr: Ug,
  detach: Zg,
  init: Wg,
  insert: Vg,
  noop: Gg,
  safe_not_equal: Hg,
  svg_element: Jg
} = window.__gradio__svelte__internal, {
  SvelteComponent: Xg,
  append: Yg,
  attr: Kg,
  detach: Qg,
  init: xg,
  insert: e2,
  noop: t2,
  safe_not_equal: n2,
  svg_element: o2
} = window.__gradio__svelte__internal, {
  SvelteComponent: na,
  create_component: oa,
  destroy_component: la,
  init: ia,
  mount_component: sa,
  safe_not_equal: aa,
  transition_in: ra,
  transition_out: _a
} = window.__gradio__svelte__internal, { createEventDispatcher: ua } = window.__gradio__svelte__internal;
function fa(n) {
  let e, t;
  return e = new at({
    props: {
      Icon: ds,
      label: (
        /*i18n*/
        n[2]("common.share")
      ),
      pending: (
        /*pending*/
        n[3]
      )
    }
  }), e.$on(
    "click",
    /*click_handler*/
    n[5]
  ), {
    c() {
      oa(e.$$.fragment);
    },
    m(o, l) {
      sa(e, o, l), t = !0;
    },
    p(o, [l]) {
      const i = {};
      l & /*i18n*/
      4 && (i.label = /*i18n*/
      o[2]("common.share")), l & /*pending*/
      8 && (i.pending = /*pending*/
      o[3]), e.$set(i);
    },
    i(o) {
      t || (ra(e.$$.fragment, o), t = !0);
    },
    o(o) {
      _a(e.$$.fragment, o), t = !1;
    },
    d(o) {
      la(e, o);
    }
  };
}
function ca(n, e, t) {
  const o = ua();
  let { formatter: l } = e, { value: i } = e, { i18n: s } = e, r = !1;
  const a = async () => {
    try {
      t(3, r = !0);
      const _ = await l(i);
      o("share", { description: _ });
    } catch (_) {
      console.error(_);
      let u = _ instanceof At ? _.message : "Share failed.";
      o("error", u);
    } finally {
      t(3, r = !1);
    }
  };
  return n.$$set = (_) => {
    "formatter" in _ && t(0, l = _.formatter), "value" in _ && t(1, i = _.value), "i18n" in _ && t(2, s = _.i18n);
  }, [l, i, s, r, o, a];
}
class da extends na {
  constructor(e) {
    super(), ia(this, e, ca, fa, aa, { formatter: 0, value: 1, i18n: 2 });
  }
}
const {
  SvelteComponent: ma,
  append: Qe,
  attr: Nn,
  create_component: pa,
  destroy_component: ha,
  detach: Rt,
  element: In,
  init: ga,
  insert: Ut,
  mount_component: ba,
  safe_not_equal: wa,
  set_data: Bn,
  space: Tn,
  text: kt,
  toggle_class: no,
  transition_in: va,
  transition_out: $a
} = window.__gradio__svelte__internal;
function oo(n) {
  let e, t, o = (
    /*i18n*/
    n[1]("common.or") + ""
  ), l, i, s, r = (
    /*message*/
    (n[2] || /*i18n*/
    n[1]("upload_text.click_to_upload")) + ""
  ), a;
  return {
    c() {
      e = In("span"), t = kt("- "), l = kt(o), i = kt(" -"), s = Tn(), a = kt(r), Nn(e, "class", "or svelte-kzcjhc");
    },
    m(_, u) {
      Ut(_, e, u), Qe(e, t), Qe(e, l), Qe(e, i), Ut(_, s, u), Ut(_, a, u);
    },
    p(_, u) {
      u & /*i18n*/
      2 && o !== (o = /*i18n*/
      _[1]("common.or") + "") && Bn(l, o), u & /*message, i18n*/
      6 && r !== (r = /*message*/
      (_[2] || /*i18n*/
      _[1]("upload_text.click_to_upload")) + "") && Bn(a, r);
    },
    d(_) {
      _ && (Rt(e), Rt(s), Rt(a));
    }
  };
}
function ka(n) {
  let e, t, o, l, i = (
    /*i18n*/
    n[1](
      /*defs*/
      n[5][
        /*type*/
        n[0]
      ] || /*defs*/
      n[5].file
    ) + ""
  ), s, r, a;
  o = new wl({});
  let _ = (
    /*mode*/
    n[3] !== "short" && oo(n)
  );
  return {
    c() {
      e = In("div"), t = In("span"), pa(o.$$.fragment), l = Tn(), s = kt(i), r = Tn(), _ && _.c(), Nn(t, "class", "icon-wrap svelte-kzcjhc"), no(
        t,
        "hovered",
        /*hovered*/
        n[4]
      ), Nn(e, "class", "wrap svelte-kzcjhc");
    },
    m(u, c) {
      Ut(u, e, c), Qe(e, t), ba(o, t, null), Qe(e, l), Qe(e, s), Qe(e, r), _ && _.m(e, null), a = !0;
    },
    p(u, [c]) {
      (!a || c & /*hovered*/
      16) && no(
        t,
        "hovered",
        /*hovered*/
        u[4]
      ), (!a || c & /*i18n, type*/
      3) && i !== (i = /*i18n*/
      u[1](
        /*defs*/
        u[5][
          /*type*/
          u[0]
        ] || /*defs*/
        u[5].file
      ) + "") && Bn(s, i), /*mode*/
      u[3] !== "short" ? _ ? _.p(u, c) : (_ = oo(u), _.c(), _.m(e, null)) : _ && (_.d(1), _ = null);
    },
    i(u) {
      a || (va(o.$$.fragment, u), a = !0);
    },
    o(u) {
      $a(o.$$.fragment, u), a = !1;
    },
    d(u) {
      u && Rt(e), ha(o), _ && _.d();
    }
  };
}
function ya(n, e, t) {
  let { type: o = "file" } = e, { i18n: l } = e, { message: i = void 0 } = e, { mode: s = "full" } = e, { hovered: r = !1 } = e;
  const a = {
    image: "upload_text.drop_image",
    video: "upload_text.drop_video",
    audio: "upload_text.drop_audio",
    file: "upload_text.drop_file",
    csv: "upload_text.drop_csv"
  };
  return n.$$set = (_) => {
    "type" in _ && t(0, o = _.type), "i18n" in _ && t(1, l = _.i18n), "message" in _ && t(2, i = _.message), "mode" in _ && t(3, s = _.mode), "hovered" in _ && t(4, r = _.hovered);
  }, [o, l, i, s, r, a];
}
class qa extends ma {
  constructor(e) {
    super(), ga(this, e, ya, ka, wa, {
      type: 0,
      i18n: 1,
      message: 2,
      mode: 3,
      hovered: 4
    });
  }
}
const {
  SvelteComponent: Sa,
  attr: Ca,
  create_slot: Ea,
  detach: Da,
  element: Ma,
  get_all_dirty_from_scope: za,
  get_slot_changes: Na,
  init: Ia,
  insert: Ba,
  safe_not_equal: Ta,
  toggle_class: lo,
  transition_in: La,
  transition_out: ja,
  update_slot_base: Fa
} = window.__gradio__svelte__internal;
function Pa(n) {
  let e, t;
  const o = (
    /*#slots*/
    n[2].default
  ), l = Ea(
    o,
    n,
    /*$$scope*/
    n[1],
    null
  );
  return {
    c() {
      e = Ma("div"), l && l.c(), Ca(e, "class", "svelte-ipfyu7"), lo(
        e,
        "show_border",
        /*show_border*/
        n[0]
      );
    },
    m(i, s) {
      Ba(i, e, s), l && l.m(e, null), t = !0;
    },
    p(i, [s]) {
      l && l.p && (!t || s & /*$$scope*/
      2) && Fa(
        l,
        o,
        i,
        /*$$scope*/
        i[1],
        t ? Na(
          o,
          /*$$scope*/
          i[1],
          s,
          null
        ) : za(
          /*$$scope*/
          i[1]
        ),
        null
      ), (!t || s & /*show_border*/
      1) && lo(
        e,
        "show_border",
        /*show_border*/
        i[0]
      );
    },
    i(i) {
      t || (La(l, i), t = !0);
    },
    o(i) {
      ja(l, i), t = !1;
    },
    d(i) {
      i && Da(e), l && l.d(i);
    }
  };
}
function Oa(n, e, t) {
  let { $$slots: o = {}, $$scope: l } = e, { show_border: i = !1 } = e;
  return n.$$set = (s) => {
    "show_border" in s && t(0, i = s.show_border), "$$scope" in s && t(1, l = s.$$scope);
  }, [i, l, o];
}
class Aa extends Sa {
  constructor(e) {
    super(), Ia(this, e, Oa, Pa, Ta, { show_border: 0 });
  }
}
const {
  SvelteComponent: l2,
  append: i2,
  attr: s2,
  check_outros: a2,
  create_component: r2,
  destroy_component: _2,
  detach: u2,
  element: f2,
  empty: c2,
  group_outros: d2,
  init: m2,
  insert: p2,
  listen: h2,
  mount_component: g2,
  safe_not_equal: b2,
  space: w2,
  toggle_class: v2,
  transition_in: $2,
  transition_out: k2
} = window.__gradio__svelte__internal, vl = (n) => {
  let e = n.currentTarget;
  const t = e.getBoundingClientRect(), o = e.naturalWidth / t.width, l = e.naturalHeight / t.height;
  if (o > l) {
    const r = e.naturalHeight / o, a = (t.height - r) / 2;
    var i = Math.round((n.clientX - t.left) * o), s = Math.round((n.clientY - t.top - a) * o);
  } else {
    const r = e.naturalWidth / l, a = (t.width - r) / 2;
    var i = Math.round((n.clientX - t.left - a) * l), s = Math.round((n.clientY - t.top) * l);
  }
  return i < 0 || i >= e.naturalWidth || s < 0 || s >= e.naturalHeight ? null : [i, s];
}, {
  SvelteComponent: Ra,
  append: io,
  attr: $e,
  bubble: so,
  check_outros: Ln,
  create_component: Et,
  destroy_component: Dt,
  detach: rt,
  element: Zt,
  empty: Ua,
  group_outros: jn,
  init: Za,
  insert: _t,
  listen: Wa,
  mount_component: Mt,
  safe_not_equal: Va,
  space: Fn,
  src_url_equal: ao,
  toggle_class: ro,
  transition_in: fe,
  transition_out: Me
} = window.__gradio__svelte__internal, { createEventDispatcher: Ga } = window.__gradio__svelte__internal;
function Ha(n) {
  let e, t, o, l, i, s, r, a, _, u = (
    /*show_download_button*/
    n[3] && _o(n)
  ), c = (
    /*show_share_button*/
    n[5] && uo(n)
  );
  return {
    c() {
      e = Zt("div"), u && u.c(), t = Fn(), c && c.c(), o = Fn(), l = Zt("button"), i = Zt("img"), $e(e, "class", "icon-buttons svelte-1e0ed51"), ao(i.src, s = /*value*/
      n[0].url) || $e(i, "src", s), $e(i, "alt", ""), $e(i, "loading", "lazy"), $e(i, "class", "svelte-1e0ed51"), ro(
        i,
        "selectable",
        /*selectable*/
        n[4]
      ), $e(l, "class", "svelte-1e0ed51");
    },
    m(d, f) {
      _t(d, e, f), u && u.m(e, null), io(e, t), c && c.m(e, null), _t(d, o, f), _t(d, l, f), io(l, i), r = !0, a || (_ = Wa(
        l,
        "click",
        /*handle_click*/
        n[7]
      ), a = !0);
    },
    p(d, f) {
      /*show_download_button*/
      d[3] ? u ? (u.p(d, f), f & /*show_download_button*/
      8 && fe(u, 1)) : (u = _o(d), u.c(), fe(u, 1), u.m(e, t)) : u && (jn(), Me(u, 1, 1, () => {
        u = null;
      }), Ln()), /*show_share_button*/
      d[5] ? c ? (c.p(d, f), f & /*show_share_button*/
      32 && fe(c, 1)) : (c = uo(d), c.c(), fe(c, 1), c.m(e, null)) : c && (jn(), Me(c, 1, 1, () => {
        c = null;
      }), Ln()), (!r || f & /*value*/
      1 && !ao(i.src, s = /*value*/
      d[0].url)) && $e(i, "src", s), (!r || f & /*selectable*/
      16) && ro(
        i,
        "selectable",
        /*selectable*/
        d[4]
      );
    },
    i(d) {
      r || (fe(u), fe(c), r = !0);
    },
    o(d) {
      Me(u), Me(c), r = !1;
    },
    d(d) {
      d && (rt(e), rt(o), rt(l)), u && u.d(), c && c.d(), a = !1, _();
    }
  };
}
function Ja(n) {
  let e, t;
  return e = new bl({
    props: {
      unpadded_box: !0,
      size: "large",
      $$slots: { default: [Xa] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      Et(e.$$.fragment);
    },
    m(o, l) {
      Mt(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l & /*$$scope*/
      4096 && (i.$$scope = { dirty: l, ctx: o }), e.$set(i);
    },
    i(o) {
      t || (fe(e.$$.fragment, o), t = !0);
    },
    o(o) {
      Me(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Dt(e, o);
    }
  };
}
function _o(n) {
  let e, t, o, l, i;
  return t = new at({
    props: {
      Icon: $s,
      label: (
        /*i18n*/
        n[6]("common.download")
      )
    }
  }), {
    c() {
      e = Zt("a"), Et(t.$$.fragment), $e(e, "href", o = /*value*/
      n[0].url), $e(e, "target", window.__is_colab__ ? "_blank" : null), $e(e, "download", l = /*value*/
      n[0].orig_name || "image");
    },
    m(s, r) {
      _t(s, e, r), Mt(t, e, null), i = !0;
    },
    p(s, r) {
      const a = {};
      r & /*i18n*/
      64 && (a.label = /*i18n*/
      s[6]("common.download")), t.$set(a), (!i || r & /*value*/
      1 && o !== (o = /*value*/
      s[0].url)) && $e(e, "href", o), (!i || r & /*value*/
      1 && l !== (l = /*value*/
      s[0].orig_name || "image")) && $e(e, "download", l);
    },
    i(s) {
      i || (fe(t.$$.fragment, s), i = !0);
    },
    o(s) {
      Me(t.$$.fragment, s), i = !1;
    },
    d(s) {
      s && rt(e), Dt(t);
    }
  };
}
function uo(n) {
  let e, t;
  return e = new da({
    props: {
      i18n: (
        /*i18n*/
        n[6]
      ),
      formatter: (
        /*func*/
        n[8]
      ),
      value: (
        /*value*/
        n[0]
      )
    }
  }), e.$on(
    "share",
    /*share_handler*/
    n[9]
  ), e.$on(
    "error",
    /*error_handler*/
    n[10]
  ), {
    c() {
      Et(e.$$.fragment);
    },
    m(o, l) {
      Mt(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l & /*i18n*/
      64 && (i.i18n = /*i18n*/
      o[6]), l & /*value*/
      1 && (i.value = /*value*/
      o[0]), e.$set(i);
    },
    i(o) {
      t || (fe(e.$$.fragment, o), t = !0);
    },
    o(o) {
      Me(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Dt(e, o);
    }
  };
}
function Xa(n) {
  let e, t;
  return e = new Yt({}), {
    c() {
      Et(e.$$.fragment);
    },
    m(o, l) {
      Mt(e, o, l), t = !0;
    },
    i(o) {
      t || (fe(e.$$.fragment, o), t = !0);
    },
    o(o) {
      Me(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Dt(e, o);
    }
  };
}
function Ya(n) {
  let e, t, o, l, i, s;
  e = new pl({
    props: {
      show_label: (
        /*show_label*/
        n[2]
      ),
      Icon: Yt,
      label: (
        /*label*/
        n[1] || /*i18n*/
        n[6]("image.image")
      )
    }
  });
  const r = [Ja, Ha], a = [];
  function _(u, c) {
    return (
      /*value*/
      u[0] === null || !/*value*/
      u[0].url ? 0 : 1
    );
  }
  return o = _(n), l = a[o] = r[o](n), {
    c() {
      Et(e.$$.fragment), t = Fn(), l.c(), i = Ua();
    },
    m(u, c) {
      Mt(e, u, c), _t(u, t, c), a[o].m(u, c), _t(u, i, c), s = !0;
    },
    p(u, [c]) {
      const d = {};
      c & /*show_label*/
      4 && (d.show_label = /*show_label*/
      u[2]), c & /*label, i18n*/
      66 && (d.label = /*label*/
      u[1] || /*i18n*/
      u[6]("image.image")), e.$set(d);
      let f = o;
      o = _(u), o === f ? a[o].p(u, c) : (jn(), Me(a[f], 1, 1, () => {
        a[f] = null;
      }), Ln(), l = a[o], l ? l.p(u, c) : (l = a[o] = r[o](u), l.c()), fe(l, 1), l.m(i.parentNode, i));
    },
    i(u) {
      s || (fe(e.$$.fragment, u), fe(l), s = !0);
    },
    o(u) {
      Me(e.$$.fragment, u), Me(l), s = !1;
    },
    d(u) {
      u && (rt(t), rt(i)), Dt(e, u), a[o].d(u);
    }
  };
}
function Ka(n, e, t) {
  let { value: o } = e, { label: l = void 0 } = e, { show_label: i } = e, { show_download_button: s = !0 } = e, { selectable: r = !1 } = e, { show_share_button: a = !1 } = e, { i18n: _ } = e;
  const u = Ga(), c = (g) => {
    let S = vl(g);
    S && u("select", { index: S, value: null });
  }, d = async (g) => g ? `<img src="${await Vl(g)}" />` : "";
  function f(g) {
    so.call(this, n, g);
  }
  function p(g) {
    so.call(this, n, g);
  }
  return n.$$set = (g) => {
    "value" in g && t(0, o = g.value), "label" in g && t(1, l = g.label), "show_label" in g && t(2, i = g.show_label), "show_download_button" in g && t(3, s = g.show_download_button), "selectable" in g && t(4, r = g.selectable), "show_share_button" in g && t(5, a = g.show_share_button), "i18n" in g && t(6, _ = g.i18n);
  }, [
    o,
    l,
    i,
    s,
    r,
    a,
    _,
    c,
    d,
    f,
    p
  ];
}
class Qa extends Ra {
  constructor(e) {
    super(), Za(this, e, Ka, Ya, Va, {
      value: 0,
      label: 1,
      show_label: 2,
      show_download_button: 3,
      selectable: 4,
      show_share_button: 5,
      i18n: 6
    });
  }
}
var pn = new Intl.Collator(0, { numeric: 1 }).compare;
function fo(n, e, t) {
  return n = n.split("."), e = e.split("."), pn(n[0], e[0]) || pn(n[1], e[1]) || (e[2] = e.slice(2).join("."), t = /[.-]/.test(n[2] = n.slice(2).join(".")), t == /[.-]/.test(e[2]) ? pn(n[2], e[2]) : t ? -1 : 1);
}
function We(n, e, t) {
  return e.startsWith("http://") || e.startsWith("https://") ? t ? n : e : n + e;
}
function hn(n) {
  if (n.startsWith("http")) {
    const { protocol: e, host: t } = new URL(n);
    return t.endsWith("hf.space") ? {
      ws_protocol: "wss",
      host: t,
      http_protocol: e
    } : {
      ws_protocol: e === "https:" ? "wss" : "ws",
      http_protocol: e,
      host: t
    };
  } else if (n.startsWith("file:"))
    return {
      ws_protocol: "ws",
      http_protocol: "http:",
      host: "lite.local"
      // Special fake hostname only used for this case. This matches the hostname allowed in `is_self_host()` in `js/wasm/network/host.ts`.
    };
  return {
    ws_protocol: "wss",
    http_protocol: "https:",
    host: n
  };
}
const $l = /^[^\/]*\/[^\/]*$/, xa = /.*hf\.space\/{0,1}$/;
async function er(n, e) {
  const t = {};
  e && (t.Authorization = `Bearer ${e}`);
  const o = n.trim();
  if ($l.test(o))
    try {
      const l = await fetch(
        `https://huggingface.co/api/spaces/${o}/host`,
        { headers: t }
      );
      if (l.status !== 200)
        throw new Error("Space metadata could not be loaded.");
      const i = (await l.json()).host;
      return {
        space_id: n,
        ...hn(i)
      };
    } catch (l) {
      throw new Error("Space metadata could not be loaded." + l.message);
    }
  if (xa.test(o)) {
    const { ws_protocol: l, http_protocol: i, host: s } = hn(o);
    return {
      space_id: s.replace(".hf.space", ""),
      ws_protocol: l,
      http_protocol: i,
      host: s
    };
  }
  return {
    space_id: !1,
    ...hn(o)
  };
}
function tr(n) {
  let e = {};
  return n.forEach(({ api_name: t }, o) => {
    t && (e[t] = o);
  }), e;
}
const nr = /^(?=[^]*\b[dD]iscussions{0,1}\b)(?=[^]*\b[dD]isabled\b)[^]*$/;
async function co(n) {
  try {
    const t = (await fetch(
      `https://huggingface.co/api/spaces/${n}/discussions`,
      {
        method: "HEAD"
      }
    )).headers.get("x-error-message");
    return !(t && nr.test(t));
  } catch {
    return !1;
  }
}
function Te(n, e, t) {
  if (n == null)
    return null;
  if (Array.isArray(n)) {
    const o = [];
    for (const l of n)
      l == null ? o.push(null) : o.push(Te(l, e, t));
    return o;
  }
  return n.is_stream ? t == null ? new ut({
    ...n,
    url: e + "/stream/" + n.path
  }) : new ut({
    ...n,
    url: "/proxy=" + t + "stream/" + n.path
  }) : new ut({
    ...n,
    url: lr(n.path, e, t)
  });
}
function or(n) {
  try {
    const e = new URL(n);
    return e.protocol === "http:" || e.protocol === "https:";
  } catch {
    return !1;
  }
}
function lr(n, e, t) {
  return n == null ? t ? `/proxy=${t}file=` : `${e}/file=` : or(n) ? n : t ? `/proxy=${t}file=${n}` : `${e}/file=${n}`;
}
async function ir(n, e, t, o = _r) {
  let l = (Array.isArray(n) ? n : [n]).map(
    (i) => i.blob
  );
  return await Promise.all(
    await o(e, l, void 0, t).then(
      async (i) => {
        if (i.error)
          throw new Error(i.error);
        return i.files ? i.files.map((s, r) => {
          const a = new ut({ ...n[r], path: s });
          return Te(a, e, null);
        }) : [];
      }
    )
  );
}
async function sr(n, e) {
  return n.map(
    (t, o) => new ut({
      path: t.name,
      orig_name: t.name,
      blob: t,
      size: t.size,
      mime_type: t.type,
      is_stream: e
    })
  );
}
class ut {
  constructor({
    path: e,
    url: t,
    orig_name: o,
    size: l,
    blob: i,
    is_stream: s,
    mime_type: r,
    alt_text: a
  }) {
    this.path = e, this.url = t, this.orig_name = o, this.size = l, this.blob = t ? void 0 : i, this.is_stream = s, this.mime_type = r, this.alt_text = a;
  }
}
const ar = "This application is too busy. Keep trying!", $t = "Connection errored out.";
let kl;
function rr(n, e) {
  return { post_data: t, upload_files: o, client: l, handle_blob: i };
  async function t(s, r, a) {
    const _ = { "Content-Type": "application/json" };
    a && (_.Authorization = `Bearer ${a}`);
    try {
      var u = await n(s, {
        method: "POST",
        body: JSON.stringify(r),
        headers: _
      });
    } catch {
      return [{ error: $t }, 500];
    }
    return [await u.json(), u.status];
  }
  async function o(s, r, a, _) {
    const u = {};
    a && (u.Authorization = `Bearer ${a}`);
    const c = 1e3, d = [];
    for (let p = 0; p < r.length; p += c) {
      const g = r.slice(p, p + c), S = new FormData();
      g.forEach((y) => {
        S.append("files", y);
      });
      try {
        const y = _ ? `${s}/upload?upload_id=${_}` : `${s}/upload`;
        var f = await n(y, {
          method: "POST",
          body: S,
          headers: u
        });
      } catch {
        return { error: $t };
      }
      const b = await f.json();
      d.push(...b);
    }
    return { files: d };
  }
  async function l(s, r = { normalise_files: !0 }) {
    return new Promise(async (a) => {
      const { status_callback: _, hf_token: u, normalise_files: c } = r, d = {
        predict: j,
        submit: le,
        view_api: ne,
        component_server: re
      }, f = c ?? !0;
      if ((typeof window > "u" || !("WebSocket" in window)) && !global.Websocket) {
        const M = await import("./wrapper-6f348d45-DOT-BUTT.js");
        kl = (await import("./__vite-browser-external-DYxpcVy9.js")).Blob, global.WebSocket = M.WebSocket;
      }
      const { ws_protocol: p, http_protocol: g, host: S, space_id: b } = await er(s, u), y = Math.random().toString(36).substring(2), h = {};
      let w, C = {}, T = !1;
      u && b && (T = await fr(b, u));
      async function L(M) {
        if (w = M, C = tr(M?.dependencies || []), w.auth_required)
          return {
            config: w,
            ...d
          };
        try {
          A = await ne(w);
        } catch (F) {
          console.error(`Could not get api details: ${F.message}`);
        }
        return {
          config: w,
          ...d
        };
      }
      let A;
      async function ee(M) {
        if (_ && _(M), M.status === "running")
          try {
            w = await go(
              n,
              `${g}//${S}`,
              u
            );
            const F = await L(w);
            a(F);
          } catch (F) {
            console.error(F), _ && _({
              status: "error",
              message: "Could not load this space.",
              load_status: "error",
              detail: "NOT_FOUND"
            });
          }
      }
      try {
        w = await go(
          n,
          `${g}//${S}`,
          u
        );
        const M = await L(w);
        a(M);
      } catch (M) {
        console.error(M), b ? On(
          b,
          $l.test(b) ? "space_name" : "subdomain",
          ee
        ) : _ && _({
          status: "error",
          message: "Could not load this space.",
          load_status: "error",
          detail: "NOT_FOUND"
        });
      }
      function j(M, F, G) {
        let Z = !1, E = !1, V;
        if (typeof M == "number")
          V = w.dependencies[M];
        else {
          const q = M.replace(/^\//, "");
          V = w.dependencies[C[q]];
        }
        if (V.types.continuous)
          throw new Error(
            "Cannot call predict on this function as it may run forever. Use submit instead"
          );
        return new Promise((q, k) => {
          const v = le(M, F, G);
          let m;
          v.on("data", (D) => {
            E && (v.destroy(), q(D)), Z = !0, m = D;
          }).on("status", (D) => {
            D.stage === "error" && k(D), D.stage === "complete" && (E = !0, Z && (v.destroy(), q(m)));
          });
        });
      }
      function le(M, F, G, Z = null) {
        let E, V;
        if (typeof M == "number")
          E = M, V = A.unnamed_endpoints[E];
        else {
          const J = M.replace(/^\//, "");
          E = C[J], V = A.named_endpoints[M.trim()];
        }
        if (typeof E != "number")
          throw new Error(
            "There is no endpoint matching that name of fn_index matching that number."
          );
        let q, k, v = w.protocol ?? "sse";
        const m = typeof M == "number" ? "/predict" : M;
        let D, z = null, P = !1;
        const W = {};
        let R = "";
        typeof window < "u" && (R = new URLSearchParams(window.location.search).toString()), i(
          `${g}//${We(S, w.path, !0)}`,
          F,
          V,
          u
        ).then((J) => {
          if (D = { data: J || [], event_data: G, fn_index: E, trigger_id: Z }, cr(E, w))
            B({
              type: "status",
              endpoint: m,
              stage: "pending",
              queue: !1,
              fn_index: E,
              time: /* @__PURE__ */ new Date()
            }), t(
              `${g}//${We(S, w.path, !0)}/run${m.startsWith("/") ? m : `/${m}`}${R ? "?" + R : ""}`,
              {
                ...D,
                session_hash: y
              },
              u
            ).then(([K, oe]) => {
              const Oe = f ? gn(
                K.data,
                V,
                w.root,
                w.root_url
              ) : K.data;
              oe == 200 ? (B({
                type: "data",
                endpoint: m,
                fn_index: E,
                data: Oe,
                time: /* @__PURE__ */ new Date()
              }), B({
                type: "status",
                endpoint: m,
                fn_index: E,
                stage: "complete",
                eta: K.average_duration,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              })) : B({
                type: "status",
                stage: "error",
                endpoint: m,
                fn_index: E,
                message: K.error,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              });
            }).catch((K) => {
              B({
                type: "status",
                stage: "error",
                message: K.message,
                endpoint: m,
                fn_index: E,
                queue: !1,
                time: /* @__PURE__ */ new Date()
              });
            });
          else if (v == "ws") {
            B({
              type: "status",
              stage: "pending",
              queue: !0,
              endpoint: m,
              fn_index: E,
              time: /* @__PURE__ */ new Date()
            });
            let K = new URL(`${p}://${We(
              S,
              w.path,
              !0
            )}
							/queue/join${R ? "?" + R : ""}`);
            T && K.searchParams.set("__sign", T), q = e(K), q.onclose = (oe) => {
              oe.wasClean || B({
                type: "status",
                stage: "error",
                broken: !0,
                message: $t,
                queue: !0,
                endpoint: m,
                fn_index: E,
                time: /* @__PURE__ */ new Date()
              });
            }, q.onmessage = function(oe) {
              const Oe = JSON.parse(oe.data), { type: pe, status: te, data: ye } = bo(
                Oe,
                h[E]
              );
              if (pe === "update" && te && !P)
                B({
                  type: "status",
                  endpoint: m,
                  fn_index: E,
                  time: /* @__PURE__ */ new Date(),
                  ...te
                }), te.stage === "error" && q.close();
              else if (pe === "hash") {
                q.send(JSON.stringify({ fn_index: E, session_hash: y }));
                return;
              } else pe === "data" ? q.send(JSON.stringify({ ...D, session_hash: y })) : pe === "complete" ? P = te : pe === "log" ? B({
                type: "log",
                log: ye.log,
                level: ye.level,
                endpoint: m,
                fn_index: E
              }) : pe === "generating" && B({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...te,
                stage: te?.stage,
                queue: !0,
                endpoint: m,
                fn_index: E
              });
              ye && (B({
                type: "data",
                time: /* @__PURE__ */ new Date(),
                data: f ? gn(
                  ye.data,
                  V,
                  w.root,
                  w.root_url
                ) : ye.data,
                endpoint: m,
                fn_index: E
              }), P && (B({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...P,
                stage: te?.stage,
                queue: !0,
                endpoint: m,
                fn_index: E
              }), q.close()));
            }, fo(w.version || "2.0.0", "3.6") < 0 && addEventListener(
              "open",
              () => q.send(JSON.stringify({ hash: y }))
            );
          } else {
            B({
              type: "status",
              stage: "pending",
              queue: !0,
              endpoint: m,
              fn_index: E,
              time: /* @__PURE__ */ new Date()
            });
            var me = new URLSearchParams({
              fn_index: E.toString(),
              session_hash: y
            }).toString();
            let K = new URL(
              `${g}//${We(
                S,
                w.path,
                !0
              )}/queue/join?${R ? R + "&" : ""}${me}`
            );
            k = new EventSource(K), k.onmessage = async function(oe) {
              const Oe = JSON.parse(oe.data), { type: pe, status: te, data: ye } = bo(
                Oe,
                h[E]
              );
              if (pe === "update" && te && !P)
                B({
                  type: "status",
                  endpoint: m,
                  fn_index: E,
                  time: /* @__PURE__ */ new Date(),
                  ...te
                }), te.stage === "error" && k.close();
              else if (pe === "data") {
                z = Oe.event_id;
                let [yf, Zl] = await t(
                  `${g}//${We(
                    S,
                    w.path,
                    !0
                  )}/queue/data`,
                  {
                    ...D,
                    session_hash: y,
                    event_id: z
                  },
                  u
                );
                Zl !== 200 && (B({
                  type: "status",
                  stage: "error",
                  message: $t,
                  queue: !0,
                  endpoint: m,
                  fn_index: E,
                  time: /* @__PURE__ */ new Date()
                }), k.close());
              } else pe === "complete" ? P = te : pe === "log" ? B({
                type: "log",
                log: ye.log,
                level: ye.level,
                endpoint: m,
                fn_index: E
              }) : pe === "generating" && B({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...te,
                stage: te?.stage,
                queue: !0,
                endpoint: m,
                fn_index: E
              });
              ye && (B({
                type: "data",
                time: /* @__PURE__ */ new Date(),
                data: f ? gn(
                  ye.data,
                  V,
                  w.root,
                  w.root_url
                ) : ye.data,
                endpoint: m,
                fn_index: E
              }), P && (B({
                type: "status",
                time: /* @__PURE__ */ new Date(),
                ...P,
                stage: te?.stage,
                queue: !0,
                endpoint: m,
                fn_index: E
              }), k.close()));
            };
          }
        });
        function B(J) {
          const K = W[J.type] || [];
          K?.forEach((oe) => oe(J));
        }
        function $(J, me) {
          const K = W, oe = K[J] || [];
          return K[J] = oe, oe?.push(me), { on: $, off: U, cancel: Y, destroy: ae };
        }
        function U(J, me) {
          const K = W;
          let oe = K[J] || [];
          return oe = oe?.filter((Oe) => Oe !== me), K[J] = oe, { on: $, off: U, cancel: Y, destroy: ae };
        }
        async function Y() {
          const J = {
            stage: "complete",
            queue: !1,
            time: /* @__PURE__ */ new Date()
          };
          P = J, B({
            ...J,
            type: "status",
            endpoint: m,
            fn_index: E
          });
          let me = {};
          v === "ws" ? (q && q.readyState === 0 ? q.addEventListener("open", () => {
            q.close();
          }) : q.close(), me = { fn_index: E, session_hash: y }) : (k.close(), me = { event_id: z });
          try {
            await n(
              `${g}//${We(
                S,
                w.path,
                !0
              )}/reset`,
              {
                headers: { "Content-Type": "application/json" },
                method: "POST",
                body: JSON.stringify(me)
              }
            );
          } catch {
            console.warn(
              "The `/reset` endpoint could not be called. Subsequent endpoint results may be unreliable."
            );
          }
        }
        function ae() {
          for (const J in W)
            W[J].forEach((me) => {
              U(J, me);
            });
        }
        return {
          on: $,
          off: U,
          cancel: Y,
          destroy: ae
        };
      }
      async function re(M, F, G) {
        var Z;
        const E = { "Content-Type": "application/json" };
        u && (E.Authorization = `Bearer ${u}`);
        let V, q = w.components.find(
          (m) => m.id === M
        );
        (Z = q?.props) != null && Z.root_url ? V = q.props.root_url : V = `${g}//${We(
          S,
          w.path,
          !0
        )}/`;
        const k = await n(
          `${V}component_server/`,
          {
            method: "POST",
            body: JSON.stringify({
              data: G,
              component_id: M,
              fn_name: F,
              session_hash: y
            }),
            headers: E
          }
        );
        if (!k.ok)
          throw new Error(
            "Could not connect to component server: " + k.statusText
          );
        return await k.json();
      }
      async function ne(M) {
        if (A)
          return A;
        const F = { "Content-Type": "application/json" };
        u && (F.Authorization = `Bearer ${u}`);
        let G;
        if (fo(M.version || "2.0.0", "3.30") < 0 ? G = await n(
          "https://gradio-space-api-fetcher-v2.hf.space/api",
          {
            method: "POST",
            body: JSON.stringify({
              serialize: !1,
              config: JSON.stringify(M)
            }),
            headers: F
          }
        ) : G = await n(`${M.root}/info`, {
          headers: F
        }), !G.ok)
          throw new Error($t);
        let Z = await G.json();
        return "api" in Z && (Z = Z.api), Z.named_endpoints["/predict"] && !Z.unnamed_endpoints[0] && (Z.unnamed_endpoints[0] = Z.named_endpoints["/predict"]), ur(Z, M, C);
      }
    });
  }
  async function i(s, r, a, _) {
    const u = await Pn(
      r,
      void 0,
      [],
      !0,
      a
    );
    return Promise.all(
      u.map(async ({ path: c, blob: d, type: f }) => {
        if (d) {
          const p = (await o(s, [d], _)).files[0];
          return { path: c, file_url: p, type: f, name: d?.name };
        }
        return { path: c, type: f };
      })
    ).then((c) => (c.forEach(({ path: d, file_url: f, type: p, name: g }) => {
      if (p === "Gallery")
        ho(r, f, d);
      else if (f) {
        const S = new ut({ path: f, orig_name: g });
        ho(r, S, d);
      }
    }), r));
  }
}
const { post_data: y2, upload_files: _r, client: q2, handle_blob: S2 } = rr(
  fetch,
  (...n) => new WebSocket(...n)
);
function gn(n, e, t, o) {
  return n.map((l, i) => {
    var s, r, a, _;
    return ((r = (s = e?.returns) == null ? void 0 : s[i]) == null ? void 0 : r.component) === "File" ? Te(l, t, o) : ((_ = (a = e?.returns) == null ? void 0 : a[i]) == null ? void 0 : _.component) === "Gallery" ? l.map((u) => Array.isArray(u) ? [Te(u[0], t, o), u[1]] : [Te(u, t, o), null]) : typeof l == "object" && l.path ? Te(l, t, o) : l;
  });
}
function mo(n, e, t, o) {
  switch (n.type) {
    case "string":
      return "string";
    case "boolean":
      return "boolean";
    case "number":
      return "number";
  }
  if (t === "JSONSerializable" || t === "StringSerializable")
    return "any";
  if (t === "ListStringSerializable")
    return "string[]";
  if (e === "Image")
    return o === "parameter" ? "Blob | File | Buffer" : "string";
  if (t === "FileSerializable")
    return n?.type === "array" ? o === "parameter" ? "(Blob | File | Buffer)[]" : "{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}[]" : o === "parameter" ? "Blob | File | Buffer" : "{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}";
  if (t === "GallerySerializable")
    return o === "parameter" ? "[(Blob | File | Buffer), (string | null)][]" : "[{ name: string; data: string; size?: number; is_file?: boolean; orig_name?: string}, (string | null))][]";
}
function po(n, e) {
  return e === "GallerySerializable" ? "array of [file, label] tuples" : e === "ListStringSerializable" ? "array of strings" : e === "FileSerializable" ? "array of files or single file" : n.description;
}
function ur(n, e, t) {
  const o = {
    named_endpoints: {},
    unnamed_endpoints: {}
  };
  for (const l in n) {
    const i = n[l];
    for (const s in i) {
      const r = e.dependencies[s] ? s : t[s.replace("/", "")], a = i[s];
      o[l][s] = {}, o[l][s].parameters = {}, o[l][s].returns = {}, o[l][s].type = e.dependencies[r].types, o[l][s].parameters = a.parameters.map(
        ({ label: _, component: u, type: c, serializer: d }) => ({
          label: _,
          component: u,
          type: mo(c, u, d, "parameter"),
          description: po(c, d)
        })
      ), o[l][s].returns = a.returns.map(
        ({ label: _, component: u, type: c, serializer: d }) => ({
          label: _,
          component: u,
          type: mo(c, u, d, "return"),
          description: po(c, d)
        })
      );
    }
  }
  return o;
}
async function fr(n, e) {
  try {
    return (await (await fetch(`https://huggingface.co/api/spaces/${n}/jwt`, {
      headers: {
        Authorization: `Bearer ${e}`
      }
    })).json()).token || !1;
  } catch (t) {
    return console.error(t), !1;
  }
}
function ho(n, e, t) {
  for (; t.length > 1; )
    n = n[t.shift()];
  n[t.shift()] = e;
}
async function Pn(n, e = void 0, t = [], o = !1, l = void 0) {
  if (Array.isArray(n)) {
    let i = [];
    return await Promise.all(
      n.map(async (s, r) => {
        var a;
        let _ = t.slice();
        _.push(r);
        const u = await Pn(
          n[r],
          o ? ((a = l?.parameters[r]) == null ? void 0 : a.component) || void 0 : e,
          _,
          !1,
          l
        );
        i = i.concat(u);
      })
    ), i;
  } else {
    if (globalThis.Buffer && n instanceof globalThis.Buffer)
      return [
        {
          path: t,
          blob: e === "Image" ? !1 : new kl([n]),
          type: e
        }
      ];
    if (typeof n == "object") {
      let i = [];
      for (let s in n)
        if (n.hasOwnProperty(s)) {
          let r = t.slice();
          r.push(s), i = i.concat(
            await Pn(
              n[s],
              void 0,
              r,
              !1,
              l
            )
          );
        }
      return i;
    }
  }
  return [];
}
function cr(n, e) {
  var t, o, l, i;
  return !(((o = (t = e?.dependencies) == null ? void 0 : t[n]) == null ? void 0 : o.queue) === null ? e.enable_queue : (i = (l = e?.dependencies) == null ? void 0 : l[n]) != null && i.queue) || !1;
}
async function go(n, e, t) {
  const o = {};
  if (t && (o.Authorization = `Bearer ${t}`), typeof window < "u" && window.gradio_config && location.origin !== "http://localhost:9876" && !window.gradio_config.dev_mode) {
    const l = window.gradio_config.root, i = window.gradio_config;
    return i.root = We(e, i.root, !1), { ...i, path: l };
  } else if (e) {
    let l = await n(`${e}/config`, {
      headers: o
    });
    if (l.status === 200) {
      const i = await l.json();
      return i.path = i.path ?? "", i.root = e, i;
    }
    throw new Error("Could not get config.");
  }
  throw new Error("No config or app endpoint found");
}
async function On(n, e, t) {
  let o = e === "subdomain" ? `https://huggingface.co/api/spaces/by-subdomain/${n}` : `https://huggingface.co/api/spaces/${n}`, l, i;
  try {
    if (l = await fetch(o), i = l.status, i !== 200)
      throw new Error();
    l = await l.json();
  } catch {
    t({
      status: "error",
      load_status: "error",
      message: "Could not get space status",
      detail: "NOT_FOUND"
    });
    return;
  }
  if (!l || i !== 200)
    return;
  const {
    runtime: { stage: s },
    id: r
  } = l;
  switch (s) {
    case "STOPPED":
    case "SLEEPING":
      t({
        status: "sleeping",
        load_status: "pending",
        message: "Space is asleep. Waking it up...",
        detail: s
      }), setTimeout(() => {
        On(n, e, t);
      }, 1e3);
      break;
    case "PAUSED":
      t({
        status: "paused",
        load_status: "error",
        message: "This space has been paused by the author. If you would like to try this demo, consider duplicating the space.",
        detail: s,
        discussions_enabled: await co(r)
      });
      break;
    case "RUNNING":
    case "RUNNING_BUILDING":
      t({
        status: "running",
        load_status: "complete",
        message: "",
        detail: s
      });
      break;
    case "BUILDING":
      t({
        status: "building",
        load_status: "pending",
        message: "Space is building...",
        detail: s
      }), setTimeout(() => {
        On(n, e, t);
      }, 1e3);
      break;
    default:
      t({
        status: "space_error",
        load_status: "error",
        message: "This space is experiencing an issue.",
        detail: s,
        discussions_enabled: await co(r)
      });
      break;
  }
}
function bo(n, e) {
  switch (n.msg) {
    case "send_data":
      return { type: "data" };
    case "send_hash":
      return { type: "hash" };
    case "queue_full":
      return {
        type: "update",
        status: {
          queue: !0,
          message: ar,
          stage: "error",
          code: n.code,
          success: n.success
        }
      };
    case "estimation":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: e || "pending",
          code: n.code,
          size: n.queue_size,
          position: n.rank,
          eta: n.rank_eta,
          success: n.success
        }
      };
    case "progress":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: "pending",
          code: n.code,
          progress_data: n.progress_data,
          success: n.success
        }
      };
    case "log":
      return { type: "log", data: n };
    case "process_generating":
      return {
        type: "generating",
        status: {
          queue: !0,
          message: n.success ? null : n.output.error,
          stage: n.success ? "generating" : "error",
          code: n.code,
          progress_data: n.progress_data,
          eta: n.average_duration
        },
        data: n.success ? n.output : null
      };
    case "process_completed":
      return "error" in n.output ? {
        type: "update",
        status: {
          queue: !0,
          message: n.output.error,
          stage: "error",
          code: n.code,
          success: n.success
        }
      } : {
        type: "complete",
        status: {
          queue: !0,
          message: n.success ? void 0 : n.output.error,
          stage: n.success ? "complete" : "error",
          code: n.code,
          progress_data: n.progress_data,
          eta: n.output.average_duration
        },
        data: n.success ? n.output : null
      };
    case "process_starts":
      return {
        type: "update",
        status: {
          queue: !0,
          stage: "pending",
          code: n.code,
          size: n.rank,
          position: 0,
          success: n.success
        }
      };
  }
  return { type: "none", status: { stage: "error", queue: !0 } };
}
const {
  SvelteComponent: dr,
  append: _e,
  attr: Ye,
  detach: yl,
  element: Ke,
  init: mr,
  insert: ql,
  noop: wo,
  safe_not_equal: pr,
  set_data: Vt,
  set_style: bn,
  space: An,
  text: lt,
  toggle_class: vo
} = window.__gradio__svelte__internal, { onMount: hr, createEventDispatcher: gr } = window.__gradio__svelte__internal;
function $o(n) {
  let e, t, o, l, i = yt(
    /*current_file_upload*/
    n[2]
  ) + "", s, r, a, _, u = (
    /*current_file_upload*/
    n[2].orig_name + ""
  ), c;
  return {
    c() {
      e = Ke("div"), t = Ke("span"), o = Ke("div"), l = Ke("progress"), s = lt(i), a = An(), _ = Ke("span"), c = lt(u), bn(l, "visibility", "hidden"), bn(l, "height", "0"), bn(l, "width", "0"), l.value = r = yt(
        /*current_file_upload*/
        n[2]
      ), Ye(l, "max", "100"), Ye(l, "class", "svelte-12ckl9l"), Ye(o, "class", "progress-bar svelte-12ckl9l"), Ye(_, "class", "file-name svelte-12ckl9l"), Ye(e, "class", "file svelte-12ckl9l");
    },
    m(d, f) {
      ql(d, e, f), _e(e, t), _e(t, o), _e(o, l), _e(l, s), _e(e, a), _e(e, _), _e(_, c);
    },
    p(d, f) {
      f & /*current_file_upload*/
      4 && i !== (i = yt(
        /*current_file_upload*/
        d[2]
      ) + "") && Vt(s, i), f & /*current_file_upload*/
      4 && r !== (r = yt(
        /*current_file_upload*/
        d[2]
      )) && (l.value = r), f & /*current_file_upload*/
      4 && u !== (u = /*current_file_upload*/
      d[2].orig_name + "") && Vt(c, u);
    },
    d(d) {
      d && yl(e);
    }
  };
}
function br(n) {
  let e, t, o, l = (
    /*files_with_progress*/
    n[0].length + ""
  ), i, s, r = (
    /*files_with_progress*/
    n[0].length > 1 ? "files" : "file"
  ), a, _, u, c = (
    /*current_file_upload*/
    n[2] && $o(n)
  );
  return {
    c() {
      e = Ke("div"), t = Ke("span"), o = lt("Uploading "), i = lt(l), s = An(), a = lt(r), _ = lt("..."), u = An(), c && c.c(), Ye(t, "class", "uploading svelte-12ckl9l"), Ye(e, "class", "wrap svelte-12ckl9l"), vo(
        e,
        "progress",
        /*progress*/
        n[1]
      );
    },
    m(d, f) {
      ql(d, e, f), _e(e, t), _e(t, o), _e(t, i), _e(t, s), _e(t, a), _e(t, _), _e(e, u), c && c.m(e, null);
    },
    p(d, [f]) {
      f & /*files_with_progress*/
      1 && l !== (l = /*files_with_progress*/
      d[0].length + "") && Vt(i, l), f & /*files_with_progress*/
      1 && r !== (r = /*files_with_progress*/
      d[0].length > 1 ? "files" : "file") && Vt(a, r), /*current_file_upload*/
      d[2] ? c ? c.p(d, f) : (c = $o(d), c.c(), c.m(e, null)) : c && (c.d(1), c = null), f & /*progress*/
      2 && vo(
        e,
        "progress",
        /*progress*/
        d[1]
      );
    },
    i: wo,
    o: wo,
    d(d) {
      d && yl(e), c && c.d();
    }
  };
}
function yt(n) {
  return n.progress * 100 / (n.size || 0) || 0;
}
function wr(n) {
  let e = 0;
  return n.forEach((t) => {
    e += yt(t);
  }), document.documentElement.style.setProperty("--upload-progress-width", (e / n.length).toFixed(2) + "%"), e / n.length;
}
function vr(n, e, t) {
  var o = this && this.__awaiter || function(f, p, g, S) {
    function b(y) {
      return y instanceof g ? y : new g(function(h) {
        h(y);
      });
    }
    return new (g || (g = Promise))(function(y, h) {
      function w(L) {
        try {
          T(S.next(L));
        } catch (A) {
          h(A);
        }
      }
      function C(L) {
        try {
          T(S.throw(L));
        } catch (A) {
          h(A);
        }
      }
      function T(L) {
        L.done ? y(L.value) : b(L.value).then(w, C);
      }
      T((S = S.apply(f, p || [])).next());
    });
  };
  let { upload_id: l } = e, { root: i } = e, { files: s } = e, r, a = !1, _, u = s.map((f) => Object.assign(Object.assign({}, f), { progress: 0 }));
  const c = gr();
  function d(f, p) {
    t(0, u = u.map((g) => (g.orig_name === f && (g.progress += p), g)));
  }
  return hr(() => {
    r = new EventSource(`${i}/upload_progress?upload_id=${l}`), r.onmessage = function(f) {
      return o(this, void 0, void 0, function* () {
        const p = JSON.parse(f.data);
        a || t(1, a = !0), p.msg === "done" ? (r.close(), c("done")) : (t(2, _ = p), d(p.orig_name, p.chunk_size));
      });
    };
  }), n.$$set = (f) => {
    "upload_id" in f && t(3, l = f.upload_id), "root" in f && t(4, i = f.root), "files" in f && t(5, s = f.files);
  }, n.$$.update = () => {
    n.$$.dirty & /*files_with_progress*/
    1 && wr(u);
  }, [u, a, _, l, i, s];
}
class $r extends dr {
  constructor(e) {
    super(), mr(this, e, vr, br, pr, { upload_id: 3, root: 4, files: 5 });
  }
}
const {
  SvelteComponent: kr,
  append: ko,
  attr: be,
  binding_callbacks: yr,
  bubble: He,
  check_outros: qr,
  create_component: Sr,
  create_slot: Cr,
  destroy_component: Er,
  detach: Sl,
  element: yo,
  empty: Dr,
  get_all_dirty_from_scope: Mr,
  get_slot_changes: zr,
  group_outros: Nr,
  init: Ir,
  insert: Cl,
  listen: we,
  mount_component: Br,
  prevent_default: Je,
  run_all: Tr,
  safe_not_equal: Lr,
  set_style: qo,
  space: jr,
  stop_propagation: Xe,
  toggle_class: Ze,
  transition_in: Gt,
  transition_out: Ht,
  update_slot_base: Fr
} = window.__gradio__svelte__internal, { createEventDispatcher: Pr, tick: Or, getContext: Ar } = window.__gradio__svelte__internal;
function Rr(n) {
  let e, t, o, l, i, s, r, a, _, u;
  const c = (
    /*#slots*/
    n[21].default
  ), d = Cr(
    c,
    n,
    /*$$scope*/
    n[20],
    null
  );
  return {
    c() {
      e = yo("button"), d && d.c(), t = jr(), o = yo("input"), be(o, "aria-label", "file upload"), be(o, "type", "file"), be(
        o,
        "accept",
        /*filetype*/
        n[1]
      ), o.multiple = l = /*file_count*/
      n[5] === "multiple" || void 0, be(o, "webkitdirectory", i = /*file_count*/
      n[5] === "directory" || void 0), be(o, "mozdirectory", s = /*file_count*/
      n[5] === "directory" || void 0), be(o, "class", "svelte-1aq8tno"), be(e, "tabindex", r = /*hidden*/
      n[7] ? -1 : 0), be(e, "class", "svelte-1aq8tno"), Ze(
        e,
        "hidden",
        /*hidden*/
        n[7]
      ), Ze(
        e,
        "center",
        /*center*/
        n[3]
      ), Ze(
        e,
        "boundedheight",
        /*boundedheight*/
        n[2]
      ), Ze(
        e,
        "flex",
        /*flex*/
        n[4]
      ), qo(
        e,
        "height",
        /*include_sources*/
        n[8] ? "calc(100% - 40px" : "100%"
      );
    },
    m(f, p) {
      Cl(f, e, p), d && d.m(e, null), ko(e, t), ko(e, o), n[29](o), a = !0, _ || (u = [
        we(
          o,
          "change",
          /*load_files_from_upload*/
          n[14]
        ),
        we(e, "drag", Xe(Je(
          /*drag_handler*/
          n[22]
        ))),
        we(e, "dragstart", Xe(Je(
          /*dragstart_handler*/
          n[23]
        ))),
        we(e, "dragend", Xe(Je(
          /*dragend_handler*/
          n[24]
        ))),
        we(e, "dragover", Xe(Je(
          /*dragover_handler*/
          n[25]
        ))),
        we(e, "dragenter", Xe(Je(
          /*dragenter_handler*/
          n[26]
        ))),
        we(e, "dragleave", Xe(Je(
          /*dragleave_handler*/
          n[27]
        ))),
        we(e, "drop", Xe(Je(
          /*drop_handler*/
          n[28]
        ))),
        we(
          e,
          "click",
          /*open_file_upload*/
          n[9]
        ),
        we(
          e,
          "drop",
          /*loadFilesFromDrop*/
          n[15]
        ),
        we(
          e,
          "dragenter",
          /*updateDragging*/
          n[13]
        ),
        we(
          e,
          "dragleave",
          /*updateDragging*/
          n[13]
        )
      ], _ = !0);
    },
    p(f, p) {
      d && d.p && (!a || p[0] & /*$$scope*/
      1048576) && Fr(
        d,
        c,
        f,
        /*$$scope*/
        f[20],
        a ? zr(
          c,
          /*$$scope*/
          f[20],
          p,
          null
        ) : Mr(
          /*$$scope*/
          f[20]
        ),
        null
      ), (!a || p[0] & /*filetype*/
      2) && be(
        o,
        "accept",
        /*filetype*/
        f[1]
      ), (!a || p[0] & /*file_count*/
      32 && l !== (l = /*file_count*/
      f[5] === "multiple" || void 0)) && (o.multiple = l), (!a || p[0] & /*file_count*/
      32 && i !== (i = /*file_count*/
      f[5] === "directory" || void 0)) && be(o, "webkitdirectory", i), (!a || p[0] & /*file_count*/
      32 && s !== (s = /*file_count*/
      f[5] === "directory" || void 0)) && be(o, "mozdirectory", s), (!a || p[0] & /*hidden*/
      128 && r !== (r = /*hidden*/
      f[7] ? -1 : 0)) && be(e, "tabindex", r), (!a || p[0] & /*hidden*/
      128) && Ze(
        e,
        "hidden",
        /*hidden*/
        f[7]
      ), (!a || p[0] & /*center*/
      8) && Ze(
        e,
        "center",
        /*center*/
        f[3]
      ), (!a || p[0] & /*boundedheight*/
      4) && Ze(
        e,
        "boundedheight",
        /*boundedheight*/
        f[2]
      ), (!a || p[0] & /*flex*/
      16) && Ze(
        e,
        "flex",
        /*flex*/
        f[4]
      ), p[0] & /*include_sources*/
      256 && qo(
        e,
        "height",
        /*include_sources*/
        f[8] ? "calc(100% - 40px" : "100%"
      );
    },
    i(f) {
      a || (Gt(d, f), a = !0);
    },
    o(f) {
      Ht(d, f), a = !1;
    },
    d(f) {
      f && Sl(e), d && d.d(f), n[29](null), _ = !1, Tr(u);
    }
  };
}
function Ur(n) {
  let e, t;
  return e = new $r({
    props: {
      root: (
        /*root*/
        n[6]
      ),
      upload_id: (
        /*upload_id*/
        n[10]
      ),
      files: (
        /*file_data*/
        n[11]
      )
    }
  }), {
    c() {
      Sr(e.$$.fragment);
    },
    m(o, l) {
      Br(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*root*/
      64 && (i.root = /*root*/
      o[6]), l[0] & /*upload_id*/
      1024 && (i.upload_id = /*upload_id*/
      o[10]), l[0] & /*file_data*/
      2048 && (i.files = /*file_data*/
      o[11]), e.$set(i);
    },
    i(o) {
      t || (Gt(e.$$.fragment, o), t = !0);
    },
    o(o) {
      Ht(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Er(e, o);
    }
  };
}
function Zr(n) {
  let e, t, o, l;
  const i = [Ur, Rr], s = [];
  function r(a, _) {
    return (
      /*uploading*/
      a[0] ? 0 : 1
    );
  }
  return e = r(n), t = s[e] = i[e](n), {
    c() {
      t.c(), o = Dr();
    },
    m(a, _) {
      s[e].m(a, _), Cl(a, o, _), l = !0;
    },
    p(a, _) {
      let u = e;
      e = r(a), e === u ? s[e].p(a, _) : (Nr(), Ht(s[u], 1, 1, () => {
        s[u] = null;
      }), qr(), t = s[e], t ? t.p(a, _) : (t = s[e] = i[e](a), t.c()), Gt(t, 1), t.m(o.parentNode, o));
    },
    i(a) {
      l || (Gt(t), l = !0);
    },
    o(a) {
      Ht(t), l = !1;
    },
    d(a) {
      a && Sl(o), s[e].d(a);
    }
  };
}
function Wr(n, e) {
  return !n || n === "*" ? !0 : n.endsWith("/*") ? e.startsWith(n.slice(0, -1)) : n === e;
}
function Vr(n, e, t) {
  let { $$slots: o = {}, $$scope: l } = e;
  var i = this && this.__awaiter || function(k, v, m, D) {
    function z(P) {
      return P instanceof m ? P : new m(function(W) {
        W(P);
      });
    }
    return new (m || (m = Promise))(function(P, W) {
      function R(U) {
        try {
          $(D.next(U));
        } catch (Y) {
          W(Y);
        }
      }
      function B(U) {
        try {
          $(D.throw(U));
        } catch (Y) {
          W(Y);
        }
      }
      function $(U) {
        U.done ? P(U.value) : z(U.value).then(R, B);
      }
      $((D = D.apply(k, v || [])).next());
    });
  };
  let { filetype: s = null } = e, { dragging: r = !1 } = e, { boundedheight: a = !0 } = e, { center: _ = !0 } = e, { flex: u = !0 } = e, { file_count: c = "single" } = e, { disable_click: d = !1 } = e, { root: f } = e, { hidden: p = !1 } = e, { format: g = "file" } = e, { include_sources: S = !1 } = e, { uploading: b = !1 } = e, y, h;
  const w = Ar("upload_files");
  let C;
  const T = Pr();
  function L() {
    t(16, r = !r);
  }
  function A() {
    d || (t(12, C.value = "", C), C.click());
  }
  function ee(k) {
    return i(this, void 0, void 0, function* () {
      yield Or(), t(10, y = Math.random().toString(36).substring(2, 15)), t(0, b = !0);
      const v = yield ir(k, f, y, w);
      return T("load", c === "single" ? v?.[0] : v), t(0, b = !1), v || [];
    });
  }
  function j(k) {
    return i(this, void 0, void 0, function* () {
      if (!k.length)
        return;
      let v = k.map((m) => new File([m], m.name));
      return t(11, h = yield sr(v)), yield ee(h);
    });
  }
  function le(k) {
    return i(this, void 0, void 0, function* () {
      const v = k.target;
      if (v.files)
        if (g != "blob")
          yield j(Array.from(v.files));
        else {
          if (c === "single") {
            T("load", v.files[0]);
            return;
          }
          T("load", v.files);
        }
    });
  }
  function re(k) {
    return i(this, void 0, void 0, function* () {
      var v;
      if (t(16, r = !1), !(!((v = k.dataTransfer) === null || v === void 0) && v.files)) return;
      const m = Array.from(k.dataTransfer.files).filter((D) => s?.split(",").some((z) => Wr(z, D.type)) ? !0 : (T("error", `Invalid file type only ${s} allowed.`), !1));
      yield j(m);
    });
  }
  function ne(k) {
    He.call(this, n, k);
  }
  function M(k) {
    He.call(this, n, k);
  }
  function F(k) {
    He.call(this, n, k);
  }
  function G(k) {
    He.call(this, n, k);
  }
  function Z(k) {
    He.call(this, n, k);
  }
  function E(k) {
    He.call(this, n, k);
  }
  function V(k) {
    He.call(this, n, k);
  }
  function q(k) {
    yr[k ? "unshift" : "push"](() => {
      C = k, t(12, C);
    });
  }
  return n.$$set = (k) => {
    "filetype" in k && t(1, s = k.filetype), "dragging" in k && t(16, r = k.dragging), "boundedheight" in k && t(2, a = k.boundedheight), "center" in k && t(3, _ = k.center), "flex" in k && t(4, u = k.flex), "file_count" in k && t(5, c = k.file_count), "disable_click" in k && t(17, d = k.disable_click), "root" in k && t(6, f = k.root), "hidden" in k && t(7, p = k.hidden), "format" in k && t(18, g = k.format), "include_sources" in k && t(8, S = k.include_sources), "uploading" in k && t(0, b = k.uploading), "$$scope" in k && t(20, l = k.$$scope);
  }, [
    b,
    s,
    a,
    _,
    u,
    c,
    f,
    p,
    S,
    A,
    y,
    h,
    C,
    L,
    le,
    re,
    r,
    d,
    g,
    j,
    l,
    o,
    ne,
    M,
    F,
    G,
    Z,
    E,
    V,
    q
  ];
}
class Gr extends kr {
  constructor(e) {
    super(), Ir(
      this,
      e,
      Vr,
      Zr,
      Lr,
      {
        filetype: 1,
        dragging: 16,
        boundedheight: 2,
        center: 3,
        flex: 4,
        file_count: 5,
        disable_click: 17,
        root: 6,
        hidden: 7,
        format: 18,
        include_sources: 8,
        uploading: 0,
        open_file_upload: 9,
        load_files: 19
      },
      null,
      [-1, -1]
    );
  }
  get open_file_upload() {
    return this.$$.ctx[9];
  }
  get load_files() {
    return this.$$.ctx[19];
  }
}
const {
  SvelteComponent: C2,
  append: E2,
  attr: D2,
  check_outros: M2,
  create_component: z2,
  destroy_component: N2,
  detach: I2,
  element: B2,
  group_outros: T2,
  init: L2,
  insert: j2,
  mount_component: F2,
  safe_not_equal: P2,
  set_style: O2,
  space: A2,
  toggle_class: R2,
  transition_in: U2,
  transition_out: Z2
} = window.__gradio__svelte__internal, { createEventDispatcher: W2 } = window.__gradio__svelte__internal, {
  SvelteComponent: Hr,
  append: So,
  attr: Jr,
  create_component: wn,
  destroy_component: vn,
  detach: Xr,
  element: Yr,
  init: Kr,
  insert: Qr,
  mount_component: $n,
  noop: xr,
  safe_not_equal: e_,
  space: Co,
  transition_in: kn,
  transition_out: yn
} = window.__gradio__svelte__internal, { createEventDispatcher: t_ } = window.__gradio__svelte__internal;
function n_(n) {
  let e, t, o, l, i, s, r;
  return t = new at({
    props: { Icon: Xs, label: "Remove Last Box" }
  }), t.$on(
    "click",
    /*click_handler*/
    n[1]
  ), l = new at({
    props: { Icon: Ds, label: "Remove All boxes" }
  }), l.$on(
    "click",
    /*click_handler_1*/
    n[2]
  ), s = new at({
    props: { Icon: is, label: "Remove Image" }
  }), s.$on(
    "click",
    /*click_handler_2*/
    n[3]
  ), {
    c() {
      e = Yr("div"), wn(t.$$.fragment), o = Co(), wn(l.$$.fragment), i = Co(), wn(s.$$.fragment), Jr(e, "class", "svelte-1o7cyxy");
    },
    m(a, _) {
      Qr(a, e, _), $n(t, e, null), So(e, o), $n(l, e, null), So(e, i), $n(s, e, null), r = !0;
    },
    p: xr,
    i(a) {
      r || (kn(t.$$.fragment, a), kn(l.$$.fragment, a), kn(s.$$.fragment, a), r = !0);
    },
    o(a) {
      yn(t.$$.fragment, a), yn(l.$$.fragment, a), yn(s.$$.fragment, a), r = !1;
    },
    d(a) {
      a && Xr(e), vn(t), vn(l), vn(s);
    }
  };
}
function o_(n) {
  const e = t_();
  return [e, (i) => {
    e("remove_box"), i.stopPropagation();
  }, (i) => {
    e("remove_boxes"), i.stopPropagation();
  }, (i) => {
    e("remove_image"), i.stopPropagation();
  }];
}
class l_ extends Hr {
  constructor(e) {
    super(), Kr(this, e, o_, n_, e_, {});
  }
}
const {
  SvelteComponent: i_,
  append: s_,
  attr: Eo,
  binding_callbacks: Do,
  bubble: a_,
  detach: r_,
  element: Mo,
  flush: jt,
  init: __,
  insert: u_,
  listen: Ce,
  noop: qn,
  run_all: f_,
  safe_not_equal: c_,
  set_style: d_,
  stop_propagation: m_
} = window.__gradio__svelte__internal, { createEventDispatcher: p_, onDestroy: h_, onMount: g_, tick: b_ } = window.__gradio__svelte__internal;
function w_(n) {
  let e, t, o, l;
  return {
    c() {
      e = Mo("div"), t = Mo("canvas"), d_(t, "z-index", "15"), Eo(t, "class", "svelte-1mnpmgt"), Eo(e, "class", "wrap svelte-1mnpmgt");
    },
    m(i, s) {
      u_(i, e, s), s_(e, t), n[13](t), n[14](e), o || (l = [
        Ce(t, "contextmenu", v_),
        Ce(
          t,
          "mousedown",
          /*handle_draw_start*/
          n[2]
        ),
        Ce(
          t,
          "mousemove",
          /*handle_draw_move*/
          n[3]
        ),
        Ce(
          t,
          "mouseout",
          /*handle_draw_move*/
          n[3]
        ),
        Ce(
          t,
          "mouseup",
          /*handle_draw_end*/
          n[4]
        ),
        Ce(
          t,
          "touchstart",
          /*handle_draw_start*/
          n[2]
        ),
        Ce(
          t,
          "touchmove",
          /*handle_draw_move*/
          n[3]
        ),
        Ce(
          t,
          "touchend",
          /*handle_draw_end*/
          n[4]
        ),
        Ce(
          t,
          "touchcancel",
          /*handle_draw_end*/
          n[4]
        ),
        Ce(
          t,
          "blur",
          /*handle_draw_end*/
          n[4]
        ),
        Ce(t, "click", m_(
          /*click_handler*/
          n[12]
        ))
      ], o = !0);
    },
    p: qn,
    i: qn,
    o: qn,
    d(i) {
      i && r_(e), n[13](null), n[14](null), o = !1, f_(l);
    }
  };
}
const v_ = (n) => n.preventDefault();
function $_(n, e, t) {
  var o = this && this.__awaiter || function(v, m, D, z) {
    function P(W) {
      return W instanceof D ? W : new D(function(R) {
        R(W);
      });
    }
    return new (D || (D = Promise))(function(W, R) {
      function B(Y) {
        try {
          U(z.next(Y));
        } catch (ae) {
          R(ae);
        }
      }
      function $(Y) {
        try {
          U(z.throw(Y));
        } catch (ae) {
          R(ae);
        }
      }
      function U(Y) {
        Y.done ? W(Y.value) : P(Y.value).then(B, $);
      }
      U((z = z.apply(v, m || [])).next());
    });
  };
  const l = p_();
  let { width: i = 0 } = e, { height: s = 0 } = e, { natural_width: r = 0 } = e, { natural_height: a = 0 } = e, _ = [], u = [], c, d, f, p = !1, g, S, b, y, h, w = 0, C = 0, T;
  function L(v) {
    return o(this, void 0, void 0, function* () {
      yield b_(), t(1, d.width = v.width, d), t(1, d.height = v.height, d), t(1, d.style.width = `${v.width}px`, d), t(1, d.style.height = `${v.height}px`, d), t(1, d.style.marginTop = `-${v.height}px`, d);
    });
  }
  function A() {
    return o(this, void 0, void 0, function* () {
      i === w && s === C || (yield L({ width: i, height: s }), G(), setTimeout(
        () => {
          C = s, w = i;
        },
        100
      ), ee());
    });
  }
  function ee() {
    return _ = [], u = [], G(), l("change", u), !0;
  }
  function j() {
    return _.pop(), u.pop(), G(), l("change", u), !0;
  }
  g_(() => o(void 0, void 0, void 0, function* () {
    f = d.getContext("2d"), f && (f.lineJoin = "round", f.lineCap = "round", f.strokeStyle = "#000"), T = new ResizeObserver(() => {
      A();
    }), T.observe(c), F(), ee();
  })), h_(() => {
    T.unobserve(c);
  });
  function le(v) {
    const m = d.getBoundingClientRect();
    let D, z;
    if (v instanceof MouseEvent)
      D = v.clientX, z = v.clientY;
    else if (v instanceof TouchEvent)
      D = v.changedTouches[0].clientX, z = v.changedTouches[0].clientY;
    else
      return { x: S, y: b };
    return {
      x: D - m.left,
      y: z - m.top
    };
  }
  function re(v) {
    v.preventDefault(), p = !0, g = 0, v instanceof MouseEvent && (g = v.button);
    const { x: m, y: D } = le(v);
    S = m, b = D;
  }
  function ne(v) {
    v.preventDefault();
    const { x: m, y: D } = le(v);
    y = m, h = D;
  }
  function M(v) {
    if (v.preventDefault(), p) {
      const { x: m, y: D } = le(v);
      let z = Math.min(S, m), P = Math.min(b, D), W = Math.max(S, m), R = Math.max(b, D);
      _.push([z, P, W, R]);
      let B = r / i, $ = a / s, U = z == W && P == R;
      const Y = U ? g === 0 ? 1 : g === 1 ? 5 : 0 : 2;
      u.push([
        Math.round(z * B),
        Math.round(P * $),
        U ? Y : 2,
        U ? 0 : Math.round(W * B),
        U ? 0 : Math.round(R * $),
        U ? 4 : 3
      ]), l("change", u);
    }
    p = !1;
  }
  function F() {
    G(), window.requestAnimationFrame(() => {
      F();
    });
  }
  function G() {
    if (f)
      if (f.clearRect(0, 0, i, s), p && y != S && b != h) {
        let v = _.slice();
        v.push([S, b, y, h]), Z(v), E(_);
      } else
        Z(_), E(_);
  }
  function Z(v) {
    f && (f.fillStyle = "rgba(0, 0, 0, 0.1)", f.beginPath(), v.forEach((m) => {
      m[0] != m[2] && m[1] != m[3] && f.rect(m[0], m[1], m[2] - m[0], m[3] - m[1]);
    }), f.fill(), f.stroke());
  }
  function E(v) {
    f && (f.beginPath(), f.fillStyle = "rgba(255, 0, 0, 1.0)", v.forEach((m, D) => {
      if (u[D][2] == 1) {
        let z = Math.sqrt(i * s) * 0.01;
        f.moveTo(m[0] + z, m[1]), f.arc(m[0], m[1], z, 0, 2 * Math.PI, !1);
      }
    }), f.fill(), f.stroke(), f.beginPath(), f.fillStyle = "rgba(0, 0, 255, 1.0)", v.forEach((m, D) => {
      if (u[D][2] == 0) {
        let z = Math.sqrt(i * s) * 0.01;
        f.moveTo(m[0] + z, m[1]), f.arc(m[0], m[1], z, 0, 2 * Math.PI, !1);
      }
    }), f.fill(), f.stroke(), f.beginPath(), f.fillStyle = "rgba(0, 255, 0, 1.0)", v.forEach((m, D) => {
      if (u[D][2] == 5) {
        let z = Math.sqrt(i * s) * 0.01;
        f.moveTo(m[0] + z, m[1]), f.arc(m[0], m[1], z, 0, 2 * Math.PI, !1);
      }
    }), f.fill(), f.stroke());
  }
  function V(v) {
    a_.call(this, n, v);
  }
  function q(v) {
    Do[v ? "unshift" : "push"](() => {
      d = v, t(1, d);
    });
  }
  function k(v) {
    Do[v ? "unshift" : "push"](() => {
      c = v, t(0, c);
    });
  }
  return n.$$set = (v) => {
    "width" in v && t(5, i = v.width), "height" in v && t(6, s = v.height), "natural_width" in v && t(7, r = v.natural_width), "natural_height" in v && t(8, a = v.natural_height);
  }, [
    c,
    d,
    re,
    ne,
    M,
    i,
    s,
    r,
    a,
    A,
    ee,
    j,
    V,
    q,
    k
  ];
}
class k_ extends i_ {
  constructor(e) {
    super(), __(
      this,
      e,
      $_,
      w_,
      c_,
      {
        width: 5,
        height: 6,
        natural_width: 7,
        natural_height: 8,
        resize_canvas: 9,
        clear: 10,
        undo: 11
      },
      null,
      [-1, -1]
    );
  }
  get width() {
    return this.$$.ctx[5];
  }
  set width(e) {
    this.$$set({ width: e }), jt();
  }
  get height() {
    return this.$$.ctx[6];
  }
  set height(e) {
    this.$$set({ height: e }), jt();
  }
  get natural_width() {
    return this.$$.ctx[7];
  }
  set natural_width(e) {
    this.$$set({ natural_width: e }), jt();
  }
  get natural_height() {
    return this.$$.ctx[8];
  }
  set natural_height(e) {
    this.$$set({ natural_height: e }), jt();
  }
  get resize_canvas() {
    return this.$$.ctx[9];
  }
  get clear() {
    return this.$$.ctx[10];
  }
  get undo() {
    return this.$$.ctx[11];
  }
}
const {
  SvelteComponent: y_,
  add_flush_callback: zo,
  append: Ft,
  attr: Ve,
  bind: No,
  binding_callbacks: Jt,
  bubble: q_,
  check_outros: St,
  create_component: ct,
  create_slot: S_,
  destroy_component: dt,
  destroy_each: C_,
  detach: mt,
  element: Rn,
  empty: El,
  ensure_array_like: Io,
  get_all_dirty_from_scope: E_,
  get_slot_changes: D_,
  group_outros: Ct,
  init: M_,
  insert: pt,
  listen: Bo,
  mount_component: ht,
  noop: z_,
  run_all: N_,
  safe_not_equal: I_,
  space: qt,
  src_url_equal: To,
  transition_in: X,
  transition_out: ie,
  update_slot_base: B_
} = window.__gradio__svelte__internal, { createEventDispatcher: T_ } = window.__gradio__svelte__internal;
function Lo(n, e, t) {
  const o = n.slice();
  return o[33] = e[t], o;
}
function jo(n) {
  let e, t;
  return e = new l_({}), e.$on(
    "remove_box",
    /*remove_box_handler*/
    n[22]
  ), e.$on(
    "remove_boxes",
    /*remove_boxes_handler*/
    n[23]
  ), e.$on(
    "remove_image",
    /*remove_image_handler*/
    n[24]
  ), {
    c() {
      ct(e.$$.fragment);
    },
    m(o, l) {
      ht(e, o, l), t = !0;
    },
    p: z_,
    i(o) {
      t || (X(e.$$.fragment, o), t = !0);
    },
    o(o) {
      ie(e.$$.fragment, o), t = !1;
    },
    d(o) {
      dt(e, o);
    }
  };
}
function Fo(n) {
  let e;
  const t = (
    /*#slots*/
    n[21].default
  ), o = S_(
    t,
    n,
    /*$$scope*/
    n[31],
    null
  );
  return {
    c() {
      o && o.c();
    },
    m(l, i) {
      o && o.m(l, i), e = !0;
    },
    p(l, i) {
      o && o.p && (!e || i[1] & /*$$scope*/
      1) && B_(
        o,
        t,
        l,
        /*$$scope*/
        l[31],
        e ? D_(
          t,
          /*$$scope*/
          l[31],
          i,
          null
        ) : E_(
          /*$$scope*/
          l[31]
        ),
        null
      );
    },
    i(l) {
      e || (X(o, l), e = !0);
    },
    o(l) {
      ie(o, l), e = !1;
    },
    d(l) {
      o && o.d(l);
    }
  };
}
function L_(n) {
  let e, t, o = (
    /*value*/
    n[0] === null && !/*active_tool*/
    n[6] && Fo(n)
  );
  return {
    c() {
      o && o.c(), e = El();
    },
    m(l, i) {
      o && o.m(l, i), pt(l, e, i), t = !0;
    },
    p(l, i) {
      /*value*/
      l[0] === null && !/*active_tool*/
      l[6] ? o ? (o.p(l, i), i[0] & /*value, active_tool*/
      65 && X(o, 1)) : (o = Fo(l), o.c(), X(o, 1), o.m(e.parentNode, e)) : o && (Ct(), ie(o, 1, 1, () => {
        o = null;
      }), St());
    },
    i(l) {
      t || (X(o), t = !0);
    },
    o(l) {
      ie(o), t = !1;
    },
    d(l) {
      l && mt(e), o && o.d(l);
    }
  };
}
function Po(n) {
  let e, t, o, l, i, s, r, a, _ = {};
  return i = new k_({ props: _ }), n[29](i), i.$on(
    "change",
    /*handle_points_change*/
    n[14]
  ), {
    c() {
      e = Rn("img"), l = qt(), ct(i.$$.fragment), To(e.src, t = /*value*/
      n[0].url) || Ve(e, "src", t), Ve(e, "alt", o = /*value*/
      n[0].alt_text), Ve(e, "class", "svelte-1qm7xww");
    },
    m(u, c) {
      pt(u, e, c), pt(u, l, c), ht(i, u, c), s = !0, r || (a = [
        Bo(
          e,
          "click",
          /*handle_click*/
          n[16]
        ),
        Bo(
          e,
          "load",
          /*handle_image_load*/
          n[13]
        )
      ], r = !0);
    },
    p(u, c) {
      (!s || c[0] & /*value*/
      1 && !To(e.src, t = /*value*/
      u[0].url)) && Ve(e, "src", t), (!s || c[0] & /*value*/
      1 && o !== (o = /*value*/
      u[0].alt_text)) && Ve(e, "alt", o);
      const d = {};
      i.$set(d);
    },
    i(u) {
      s || (X(i.$$.fragment, u), s = !0);
    },
    o(u) {
      ie(i.$$.fragment, u), s = !1;
    },
    d(u) {
      u && (mt(e), mt(l)), n[29](null), dt(i, u), r = !1, N_(a);
    }
  };
}
function Oo(n) {
  let e, t;
  return e = new Aa({
    props: {
      show_border: !/*value*/
      n[0]?.url,
      $$slots: { default: [j_] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      ct(e.$$.fragment);
    },
    m(o, l) {
      ht(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*value*/
      1 && (i.show_border = !/*value*/
      o[0]?.url), l[0] & /*sources_list*/
      2048 | l[1] & /*$$scope*/
      1 && (i.$$scope = { dirty: l, ctx: o }), e.$set(i);
    },
    i(o) {
      t || (X(e.$$.fragment, o), t = !0);
    },
    o(o) {
      ie(e.$$.fragment, o), t = !1;
    },
    d(o) {
      dt(e, o);
    }
  };
}
function Ao(n) {
  let e, t;
  function o() {
    return (
      /*click_handler*/
      n[30](
        /*source*/
        n[33]
      )
    );
  }
  return e = new at({
    props: {
      Icon: (
        /*sources_meta*/
        n[17][
          /*source*/
          n[33]
        ].icon
      ),
      size: "large",
      label: (
        /*source*/
        n[33] + "-image-toolbar-btn"
      ),
      padded: !1
    }
  }), e.$on("click", o), {
    c() {
      ct(e.$$.fragment);
    },
    m(l, i) {
      ht(e, l, i), t = !0;
    },
    p(l, i) {
      n = l;
      const s = {};
      i[0] & /*sources_list*/
      2048 && (s.Icon = /*sources_meta*/
      n[17][
        /*source*/
        n[33]
      ].icon), i[0] & /*sources_list*/
      2048 && (s.label = /*source*/
      n[33] + "-image-toolbar-btn"), e.$set(s);
    },
    i(l) {
      t || (X(e.$$.fragment, l), t = !0);
    },
    o(l) {
      ie(e.$$.fragment, l), t = !1;
    },
    d(l) {
      dt(e, l);
    }
  };
}
function j_(n) {
  let e, t, o = Io(
    /*sources_list*/
    n[11]
  ), l = [];
  for (let s = 0; s < o.length; s += 1)
    l[s] = Ao(Lo(n, o, s));
  const i = (s) => ie(l[s], 1, 1, () => {
    l[s] = null;
  });
  return {
    c() {
      for (let s = 0; s < l.length; s += 1)
        l[s].c();
      e = El();
    },
    m(s, r) {
      for (let a = 0; a < l.length; a += 1)
        l[a] && l[a].m(s, r);
      pt(s, e, r), t = !0;
    },
    p(s, r) {
      if (r[0] & /*sources_meta, sources_list, handle_toolbar*/
      395264) {
        o = Io(
          /*sources_list*/
          s[11]
        );
        let a;
        for (a = 0; a < o.length; a += 1) {
          const _ = Lo(s, o, a);
          l[a] ? (l[a].p(_, r), X(l[a], 1)) : (l[a] = Ao(_), l[a].c(), X(l[a], 1), l[a].m(e.parentNode, e));
        }
        for (Ct(), a = o.length; a < l.length; a += 1)
          i(a);
        St();
      }
    },
    i(s) {
      if (!t) {
        for (let r = 0; r < o.length; r += 1)
          X(l[r]);
        t = !0;
      }
    },
    o(s) {
      l = l.filter(Boolean);
      for (let r = 0; r < l.length; r += 1)
        ie(l[r]);
      t = !1;
    },
    d(s) {
      s && mt(e), C_(l, s);
    }
  };
}
function F_(n) {
  let e, t, o, l, i, s, r, a, _, u, c = (
    /*sources*/
    n[3].length > 1 || /*sources*/
    n[3].includes("clipboard")
  ), d;
  e = new pl({
    props: {
      show_label: (
        /*show_label*/
        n[2]
      ),
      Icon: Yt,
      label: (
        /*label*/
        n[1] || "Image"
      )
    }
  });
  let f = (
    /*value*/
    n[0]?.url && jo(n)
  );
  function p(h) {
    n[26](h);
  }
  function g(h) {
    n[27](h);
  }
  let S = {
    hidden: (
      /*value*/
      n[0] !== null || /*active_tool*/
      n[6] === "webcam"
    ),
    filetype: "image/*",
    root: (
      /*root*/
      n[5]
    ),
    disable_click: !/*sources*/
    n[3].includes("upload"),
    $$slots: { default: [L_] },
    $$scope: { ctx: n }
  };
  /*uploading*/
  n[7] !== void 0 && (S.uploading = /*uploading*/
  n[7]), /*dragging*/
  n[8] !== void 0 && (S.dragging = /*dragging*/
  n[8]), s = new Gr({ props: S }), n[25](s), Jt.push(() => No(s, "uploading", p)), Jt.push(() => No(s, "dragging", g)), s.$on(
    "load",
    /*handle_upload*/
    n[15]
  ), s.$on(
    "error",
    /*error_handler*/
    n[28]
  );
  let b = (
    /*value*/
    n[0] !== null && !/*streaming*/
    n[4] && Po(n)
  ), y = c && Oo(n);
  return {
    c() {
      ct(e.$$.fragment), t = qt(), o = Rn("div"), f && f.c(), l = qt(), i = Rn("div"), ct(s.$$.fragment), _ = qt(), b && b.c(), u = qt(), y && y.c(), Ve(i, "class", "upload-container svelte-1qm7xww"), Ve(o, "data-testid", "image"), Ve(o, "class", "image-container svelte-1qm7xww");
    },
    m(h, w) {
      ht(e, h, w), pt(h, t, w), pt(h, o, w), f && f.m(o, null), Ft(o, l), Ft(o, i), ht(s, i, null), Ft(i, _), b && b.m(i, null), Ft(o, u), y && y.m(o, null), d = !0;
    },
    p(h, w) {
      const C = {};
      w[0] & /*show_label*/
      4 && (C.show_label = /*show_label*/
      h[2]), w[0] & /*label*/
      2 && (C.label = /*label*/
      h[1] || "Image"), e.$set(C), /*value*/
      h[0]?.url ? f ? (f.p(h, w), w[0] & /*value*/
      1 && X(f, 1)) : (f = jo(h), f.c(), X(f, 1), f.m(o, l)) : f && (Ct(), ie(f, 1, 1, () => {
        f = null;
      }), St());
      const T = {};
      w[0] & /*value, active_tool*/
      65 && (T.hidden = /*value*/
      h[0] !== null || /*active_tool*/
      h[6] === "webcam"), w[0] & /*root*/
      32 && (T.root = /*root*/
      h[5]), w[0] & /*sources*/
      8 && (T.disable_click = !/*sources*/
      h[3].includes("upload")), w[0] & /*value, active_tool*/
      65 | w[1] & /*$$scope*/
      1 && (T.$$scope = { dirty: w, ctx: h }), !r && w[0] & /*uploading*/
      128 && (r = !0, T.uploading = /*uploading*/
      h[7], zo(() => r = !1)), !a && w[0] & /*dragging*/
      256 && (a = !0, T.dragging = /*dragging*/
      h[8], zo(() => a = !1)), s.$set(T), /*value*/
      h[0] !== null && !/*streaming*/
      h[4] ? b ? (b.p(h, w), w[0] & /*value, streaming*/
      17 && X(b, 1)) : (b = Po(h), b.c(), X(b, 1), b.m(i, null)) : b && (Ct(), ie(b, 1, 1, () => {
        b = null;
      }), St()), w[0] & /*sources*/
      8 && (c = /*sources*/
      h[3].length > 1 || /*sources*/
      h[3].includes("clipboard")), c ? y ? (y.p(h, w), w[0] & /*sources*/
      8 && X(y, 1)) : (y = Oo(h), y.c(), X(y, 1), y.m(o, null)) : y && (Ct(), ie(y, 1, 1, () => {
        y = null;
      }), St());
    },
    i(h) {
      d || (X(e.$$.fragment, h), X(f), X(s.$$.fragment, h), X(b), X(y), d = !0);
    },
    o(h) {
      ie(e.$$.fragment, h), ie(f), ie(s.$$.fragment, h), ie(b), ie(y), d = !1;
    },
    d(h) {
      h && (mt(t), mt(o)), dt(e, h), f && f.d(), n[25](null), dt(s), b && b.d(), y && y.d();
    }
  };
}
function P_(n, e, t) {
  let o, { $$slots: l = {}, $$scope: i } = e;
  var s = this && this.__awaiter || function(q, k, v, m) {
    function D(z) {
      return z instanceof v ? z : new v(function(P) {
        P(z);
      });
    }
    return new (v || (v = Promise))(function(z, P) {
      function W($) {
        try {
          B(m.next($));
        } catch (U) {
          P(U);
        }
      }
      function R($) {
        try {
          B(m.throw($));
        } catch (U) {
          P(U);
        }
      }
      function B($) {
        $.done ? z($.value) : D($.value).then(W, R);
      }
      B((m = m.apply(q, k || [])).next());
    });
  };
  const r = T_();
  let a, { value: _ } = e, { points: u } = e, { label: c = void 0 } = e, { show_label: d } = e;
  function f(q) {
    const k = q.currentTarget;
    t(9, a.width = k.width, a), t(9, a.height = k.height, a), t(9, a.natural_width = k.naturalWidth, a), t(9, a.natural_height = k.naturalHeight, a), a.resize_canvas();
  }
  function p({ detail: q }) {
    t(19, u = q), r("points_change", q);
  }
  let { sources: g = ["upload", "clipboard"] } = e, { streaming: S = !1 } = e, { root: b } = e, { i18n: y } = e, h, w = !1, { active_tool: C = null } = e;
  function T({ detail: q }) {
    t(0, _ = Te(q, b, null)), r("upload", q);
  }
  let L = !1;
  function A(q) {
    let k = vl(q);
    k && r("select", { index: k, value: null });
  }
  const ee = {
    upload: {
      icon: wl,
      label: y("Upload"),
      order: 0
    },
    clipboard: {
      icon: Us,
      label: y("Paste"),
      order: 2
    }
  };
  function j(q) {
    return s(this, void 0, void 0, function* () {
      switch (q) {
        case "clipboard":
          navigator.clipboard.read().then((k) => s(this, void 0, void 0, function* () {
            for (let v = 0; v < k.length; v++) {
              const m = k[v].types.find((D) => D.startsWith("image/"));
              if (m) {
                t(0, _ = null), k[v].getType(m).then((D) => s(this, void 0, void 0, function* () {
                  const z = yield h.load_files([new File([D], `clipboard.${m.replace("image/", "")}`)]);
                  t(0, _ = z?.[0] || null);
                }));
                break;
              }
            }
          }));
          break;
        case "upload":
          h.open_file_upload();
          break;
      }
    });
  }
  const le = () => {
    a.undo();
  }, re = () => {
    a.clear();
  }, ne = () => {
    t(0, _ = null), r("clear");
  };
  function M(q) {
    Jt[q ? "unshift" : "push"](() => {
      h = q, t(10, h);
    });
  }
  function F(q) {
    w = q, t(7, w);
  }
  function G(q) {
    L = q, t(8, L);
  }
  function Z(q) {
    q_.call(this, n, q);
  }
  function E(q) {
    Jt[q ? "unshift" : "push"](() => {
      a = q, t(9, a);
    });
  }
  const V = (q) => j(q);
  return n.$$set = (q) => {
    "value" in q && t(0, _ = q.value), "points" in q && t(19, u = q.points), "label" in q && t(1, c = q.label), "show_label" in q && t(2, d = q.show_label), "sources" in q && t(3, g = q.sources), "streaming" in q && t(4, S = q.streaming), "root" in q && t(5, b = q.root), "i18n" in q && t(20, y = q.i18n), "active_tool" in q && t(6, C = q.active_tool), "$$scope" in q && t(31, i = q.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty[0] & /*uploading*/
    128 && w && t(0, _ = null), n.$$.dirty[0] & /*value, root*/
    33 && _ && !_.url && t(0, _ = Te(_, b, null)), n.$$.dirty[0] & /*dragging*/
    256 && r("drag", L), n.$$.dirty[0] & /*sources*/
    8 && t(11, o = g.sort((q, k) => ee[q].order - ee[k].order));
  }, [
    _,
    c,
    d,
    g,
    S,
    b,
    C,
    w,
    L,
    a,
    h,
    o,
    r,
    f,
    p,
    T,
    A,
    ee,
    j,
    u,
    y,
    l,
    le,
    re,
    ne,
    M,
    F,
    G,
    Z,
    E,
    V,
    i
  ];
}
class O_ extends y_ {
  constructor(e) {
    super(), M_(
      this,
      e,
      P_,
      F_,
      I_,
      {
        value: 0,
        points: 19,
        label: 1,
        show_label: 2,
        sources: 3,
        streaming: 4,
        root: 5,
        i18n: 20,
        active_tool: 6
      },
      null,
      [-1, -1]
    );
  }
}
function it(n) {
  let e = ["", "k", "M", "G", "T", "P", "E", "Z"], t = 0;
  for (; n > 1e3 && t < e.length - 1; )
    n /= 1e3, t++;
  let o = e[t];
  return (Number.isInteger(n) ? n : n.toFixed(1)) + o;
}
function Wt() {
}
function A_(n, e) {
  return n != n ? e == e : n !== e || n && typeof n == "object" || typeof n == "function";
}
const Dl = typeof window < "u";
let Ro = Dl ? () => window.performance.now() : () => Date.now(), Ml = Dl ? (n) => requestAnimationFrame(n) : Wt;
const ft = /* @__PURE__ */ new Set();
function zl(n) {
  ft.forEach((e) => {
    e.c(n) || (ft.delete(e), e.f());
  }), ft.size !== 0 && Ml(zl);
}
function R_(n) {
  let e;
  return ft.size === 0 && Ml(zl), {
    promise: new Promise((t) => {
      ft.add(e = { c: n, f: t });
    }),
    abort() {
      ft.delete(e);
    }
  };
}
const nt = [];
function U_(n, e = Wt) {
  let t;
  const o = /* @__PURE__ */ new Set();
  function l(r) {
    if (A_(n, r) && (n = r, t)) {
      const a = !nt.length;
      for (const _ of o)
        _[1](), nt.push(_, n);
      if (a) {
        for (let _ = 0; _ < nt.length; _ += 2)
          nt[_][0](nt[_ + 1]);
        nt.length = 0;
      }
    }
  }
  function i(r) {
    l(r(n));
  }
  function s(r, a = Wt) {
    const _ = [r, a];
    return o.add(_), o.size === 1 && (t = e(l, i) || Wt), r(n), () => {
      o.delete(_), o.size === 0 && t && (t(), t = null);
    };
  }
  return { set: l, update: i, subscribe: s };
}
function Uo(n) {
  return Object.prototype.toString.call(n) === "[object Date]";
}
function Un(n, e, t, o) {
  if (typeof t == "number" || Uo(t)) {
    const l = o - t, i = (t - e) / (n.dt || 1 / 60), s = n.opts.stiffness * l, r = n.opts.damping * i, a = (s - r) * n.inv_mass, _ = (i + a) * n.dt;
    return Math.abs(_) < n.opts.precision && Math.abs(l) < n.opts.precision ? o : (n.settled = !1, Uo(t) ? new Date(t.getTime() + _) : t + _);
  } else {
    if (Array.isArray(t))
      return t.map(
        (l, i) => Un(n, e[i], t[i], o[i])
      );
    if (typeof t == "object") {
      const l = {};
      for (const i in t)
        l[i] = Un(n, e[i], t[i], o[i]);
      return l;
    } else
      throw new Error(`Cannot spring ${typeof t} values`);
  }
}
function Zo(n, e = {}) {
  const t = U_(n), { stiffness: o = 0.15, damping: l = 0.8, precision: i = 0.01 } = e;
  let s, r, a, _ = n, u = n, c = 1, d = 0, f = !1;
  function p(S, b = {}) {
    u = S;
    const y = a = {};
    return n == null || b.hard || g.stiffness >= 1 && g.damping >= 1 ? (f = !0, s = Ro(), _ = S, t.set(n = u), Promise.resolve()) : (b.soft && (d = 1 / ((b.soft === !0 ? 0.5 : +b.soft) * 60), c = 0), r || (s = Ro(), f = !1, r = R_((h) => {
      if (f)
        return f = !1, r = null, !1;
      c = Math.min(c + d, 1);
      const w = {
        inv_mass: c,
        opts: g,
        settled: !0,
        dt: (h - s) * 60 / 1e3
      }, C = Un(w, _, n, u);
      return s = h, _ = n, t.set(n = C), w.settled && (r = null), !w.settled;
    })), new Promise((h) => {
      r.promise.then(() => {
        y === a && h();
      });
    }));
  }
  const g = {
    set: p,
    update: (S, b) => p(S(u, n), b),
    subscribe: t.subscribe,
    stiffness: o,
    damping: l,
    precision: i
  };
  return g;
}
const {
  SvelteComponent: Z_,
  append: Ee,
  attr: O,
  component_subscribe: Wo,
  detach: W_,
  element: V_,
  init: G_,
  insert: H_,
  noop: Vo,
  safe_not_equal: J_,
  set_style: Pt,
  svg_element: De,
  toggle_class: Go
} = window.__gradio__svelte__internal, { onMount: X_ } = window.__gradio__svelte__internal;
function Y_(n) {
  let e, t, o, l, i, s, r, a, _, u, c, d;
  return {
    c() {
      e = V_("div"), t = De("svg"), o = De("g"), l = De("path"), i = De("path"), s = De("path"), r = De("path"), a = De("g"), _ = De("path"), u = De("path"), c = De("path"), d = De("path"), O(l, "d", "M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z"), O(l, "fill", "#FF7C00"), O(l, "fill-opacity", "0.4"), O(l, "class", "svelte-43sxxs"), O(i, "d", "M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z"), O(i, "fill", "#FF7C00"), O(i, "class", "svelte-43sxxs"), O(s, "d", "M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z"), O(s, "fill", "#FF7C00"), O(s, "fill-opacity", "0.4"), O(s, "class", "svelte-43sxxs"), O(r, "d", "M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z"), O(r, "fill", "#FF7C00"), O(r, "class", "svelte-43sxxs"), Pt(o, "transform", "translate(" + /*$top*/
      n[1][0] + "px, " + /*$top*/
      n[1][1] + "px)"), O(_, "d", "M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z"), O(_, "fill", "#FF7C00"), O(_, "fill-opacity", "0.4"), O(_, "class", "svelte-43sxxs"), O(u, "d", "M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z"), O(u, "fill", "#FF7C00"), O(u, "class", "svelte-43sxxs"), O(c, "d", "M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z"), O(c, "fill", "#FF7C00"), O(c, "fill-opacity", "0.4"), O(c, "class", "svelte-43sxxs"), O(d, "d", "M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z"), O(d, "fill", "#FF7C00"), O(d, "class", "svelte-43sxxs"), Pt(a, "transform", "translate(" + /*$bottom*/
      n[2][0] + "px, " + /*$bottom*/
      n[2][1] + "px)"), O(t, "viewBox", "-1200 -1200 3000 3000"), O(t, "fill", "none"), O(t, "xmlns", "http://www.w3.org/2000/svg"), O(t, "class", "svelte-43sxxs"), O(e, "class", "svelte-43sxxs"), Go(
        e,
        "margin",
        /*margin*/
        n[0]
      );
    },
    m(f, p) {
      H_(f, e, p), Ee(e, t), Ee(t, o), Ee(o, l), Ee(o, i), Ee(o, s), Ee(o, r), Ee(t, a), Ee(a, _), Ee(a, u), Ee(a, c), Ee(a, d);
    },
    p(f, [p]) {
      p & /*$top*/
      2 && Pt(o, "transform", "translate(" + /*$top*/
      f[1][0] + "px, " + /*$top*/
      f[1][1] + "px)"), p & /*$bottom*/
      4 && Pt(a, "transform", "translate(" + /*$bottom*/
      f[2][0] + "px, " + /*$bottom*/
      f[2][1] + "px)"), p & /*margin*/
      1 && Go(
        e,
        "margin",
        /*margin*/
        f[0]
      );
    },
    i: Vo,
    o: Vo,
    d(f) {
      f && W_(e);
    }
  };
}
function K_(n, e, t) {
  let o, l;
  var i = this && this.__awaiter || function(f, p, g, S) {
    function b(y) {
      return y instanceof g ? y : new g(function(h) {
        h(y);
      });
    }
    return new (g || (g = Promise))(function(y, h) {
      function w(L) {
        try {
          T(S.next(L));
        } catch (A) {
          h(A);
        }
      }
      function C(L) {
        try {
          T(S.throw(L));
        } catch (A) {
          h(A);
        }
      }
      function T(L) {
        L.done ? y(L.value) : b(L.value).then(w, C);
      }
      T((S = S.apply(f, p || [])).next());
    });
  };
  let { margin: s = !0 } = e;
  const r = Zo([0, 0]);
  Wo(n, r, (f) => t(1, o = f));
  const a = Zo([0, 0]);
  Wo(n, a, (f) => t(2, l = f));
  let _;
  function u() {
    return i(this, void 0, void 0, function* () {
      yield Promise.all([r.set([125, 140]), a.set([-125, -140])]), yield Promise.all([r.set([-125, 140]), a.set([125, -140])]), yield Promise.all([r.set([-125, 0]), a.set([125, -0])]), yield Promise.all([r.set([125, 0]), a.set([-125, 0])]);
    });
  }
  function c() {
    return i(this, void 0, void 0, function* () {
      yield u(), _ || c();
    });
  }
  function d() {
    return i(this, void 0, void 0, function* () {
      yield Promise.all([r.set([125, 0]), a.set([-125, 0])]), c();
    });
  }
  return X_(() => (d(), () => _ = !0)), n.$$set = (f) => {
    "margin" in f && t(0, s = f.margin);
  }, [s, o, l, r, a];
}
class Q_ extends Z_ {
  constructor(e) {
    super(), G_(this, e, K_, Y_, J_, { margin: 0 });
  }
}
const {
  SvelteComponent: x_,
  append: xe,
  attr: Ne,
  binding_callbacks: Ho,
  check_outros: Nl,
  create_component: eu,
  create_slot: tu,
  destroy_component: nu,
  destroy_each: Il,
  detach: N,
  element: Le,
  empty: wt,
  ensure_array_like: Xt,
  get_all_dirty_from_scope: ou,
  get_slot_changes: lu,
  group_outros: Bl,
  init: iu,
  insert: I,
  mount_component: su,
  noop: Zn,
  safe_not_equal: au,
  set_data: ke,
  set_style: Ge,
  space: Ie,
  text: H,
  toggle_class: ve,
  transition_in: gt,
  transition_out: bt,
  update_slot_base: ru
} = window.__gradio__svelte__internal, { tick: _u } = window.__gradio__svelte__internal, { onDestroy: uu } = window.__gradio__svelte__internal, fu = (n) => ({}), Jo = (n) => ({});
function Xo(n, e, t) {
  const o = n.slice();
  return o[39] = e[t], o[41] = t, o;
}
function Yo(n, e, t) {
  const o = n.slice();
  return o[39] = e[t], o;
}
function cu(n) {
  let e, t = (
    /*i18n*/
    n[1]("common.error") + ""
  ), o, l, i;
  const s = (
    /*#slots*/
    n[29].error
  ), r = tu(
    s,
    n,
    /*$$scope*/
    n[28],
    Jo
  );
  return {
    c() {
      e = Le("span"), o = H(t), l = Ie(), r && r.c(), Ne(e, "class", "error svelte-1txqlrd");
    },
    m(a, _) {
      I(a, e, _), xe(e, o), I(a, l, _), r && r.m(a, _), i = !0;
    },
    p(a, _) {
      (!i || _[0] & /*i18n*/
      2) && t !== (t = /*i18n*/
      a[1]("common.error") + "") && ke(o, t), r && r.p && (!i || _[0] & /*$$scope*/
      268435456) && ru(
        r,
        s,
        a,
        /*$$scope*/
        a[28],
        i ? lu(
          s,
          /*$$scope*/
          a[28],
          _,
          fu
        ) : ou(
          /*$$scope*/
          a[28]
        ),
        Jo
      );
    },
    i(a) {
      i || (gt(r, a), i = !0);
    },
    o(a) {
      bt(r, a), i = !1;
    },
    d(a) {
      a && (N(e), N(l)), r && r.d(a);
    }
  };
}
function du(n) {
  let e, t, o, l, i, s, r, a, _, u = (
    /*variant*/
    n[8] === "default" && /*show_eta_bar*/
    n[18] && /*show_progress*/
    n[6] === "full" && Ko(n)
  );
  function c(h, w) {
    if (
      /*progress*/
      h[7]
    ) return hu;
    if (
      /*queue_position*/
      h[2] !== null && /*queue_size*/
      h[3] !== void 0 && /*queue_position*/
      h[2] >= 0
    ) return pu;
    if (
      /*queue_position*/
      h[2] === 0
    ) return mu;
  }
  let d = c(n), f = d && d(n), p = (
    /*timer*/
    n[5] && el(n)
  );
  const g = [vu, wu], S = [];
  function b(h, w) {
    return (
      /*last_progress_level*/
      h[15] != null ? 0 : (
        /*show_progress*/
        h[6] === "full" ? 1 : -1
      )
    );
  }
  ~(i = b(n)) && (s = S[i] = g[i](n));
  let y = !/*timer*/
  n[5] && al(n);
  return {
    c() {
      u && u.c(), e = Ie(), t = Le("div"), f && f.c(), o = Ie(), p && p.c(), l = Ie(), s && s.c(), r = Ie(), y && y.c(), a = wt(), Ne(t, "class", "progress-text svelte-1txqlrd"), ve(
        t,
        "meta-text-center",
        /*variant*/
        n[8] === "center"
      ), ve(
        t,
        "meta-text",
        /*variant*/
        n[8] === "default"
      );
    },
    m(h, w) {
      u && u.m(h, w), I(h, e, w), I(h, t, w), f && f.m(t, null), xe(t, o), p && p.m(t, null), I(h, l, w), ~i && S[i].m(h, w), I(h, r, w), y && y.m(h, w), I(h, a, w), _ = !0;
    },
    p(h, w) {
      /*variant*/
      h[8] === "default" && /*show_eta_bar*/
      h[18] && /*show_progress*/
      h[6] === "full" ? u ? u.p(h, w) : (u = Ko(h), u.c(), u.m(e.parentNode, e)) : u && (u.d(1), u = null), d === (d = c(h)) && f ? f.p(h, w) : (f && f.d(1), f = d && d(h), f && (f.c(), f.m(t, o))), /*timer*/
      h[5] ? p ? p.p(h, w) : (p = el(h), p.c(), p.m(t, null)) : p && (p.d(1), p = null), (!_ || w[0] & /*variant*/
      256) && ve(
        t,
        "meta-text-center",
        /*variant*/
        h[8] === "center"
      ), (!_ || w[0] & /*variant*/
      256) && ve(
        t,
        "meta-text",
        /*variant*/
        h[8] === "default"
      );
      let C = i;
      i = b(h), i === C ? ~i && S[i].p(h, w) : (s && (Bl(), bt(S[C], 1, 1, () => {
        S[C] = null;
      }), Nl()), ~i ? (s = S[i], s ? s.p(h, w) : (s = S[i] = g[i](h), s.c()), gt(s, 1), s.m(r.parentNode, r)) : s = null), /*timer*/
      h[5] ? y && (y.d(1), y = null) : y ? y.p(h, w) : (y = al(h), y.c(), y.m(a.parentNode, a));
    },
    i(h) {
      _ || (gt(s), _ = !0);
    },
    o(h) {
      bt(s), _ = !1;
    },
    d(h) {
      h && (N(e), N(t), N(l), N(r), N(a)), u && u.d(h), f && f.d(), p && p.d(), ~i && S[i].d(h), y && y.d(h);
    }
  };
}
function Ko(n) {
  let e, t = `translateX(${/*eta_level*/
  (n[17] || 0) * 100 - 100}%)`;
  return {
    c() {
      e = Le("div"), Ne(e, "class", "eta-bar svelte-1txqlrd"), Ge(e, "transform", t);
    },
    m(o, l) {
      I(o, e, l);
    },
    p(o, l) {
      l[0] & /*eta_level*/
      131072 && t !== (t = `translateX(${/*eta_level*/
      (o[17] || 0) * 100 - 100}%)`) && Ge(e, "transform", t);
    },
    d(o) {
      o && N(e);
    }
  };
}
function mu(n) {
  let e;
  return {
    c() {
      e = H("processing |");
    },
    m(t, o) {
      I(t, e, o);
    },
    p: Zn,
    d(t) {
      t && N(e);
    }
  };
}
function pu(n) {
  let e, t = (
    /*queue_position*/
    n[2] + 1 + ""
  ), o, l, i, s;
  return {
    c() {
      e = H("queue: "), o = H(t), l = H("/"), i = H(
        /*queue_size*/
        n[3]
      ), s = H(" |");
    },
    m(r, a) {
      I(r, e, a), I(r, o, a), I(r, l, a), I(r, i, a), I(r, s, a);
    },
    p(r, a) {
      a[0] & /*queue_position*/
      4 && t !== (t = /*queue_position*/
      r[2] + 1 + "") && ke(o, t), a[0] & /*queue_size*/
      8 && ke(
        i,
        /*queue_size*/
        r[3]
      );
    },
    d(r) {
      r && (N(e), N(o), N(l), N(i), N(s));
    }
  };
}
function hu(n) {
  let e, t = Xt(
    /*progress*/
    n[7]
  ), o = [];
  for (let l = 0; l < t.length; l += 1)
    o[l] = xo(Yo(n, t, l));
  return {
    c() {
      for (let l = 0; l < o.length; l += 1)
        o[l].c();
      e = wt();
    },
    m(l, i) {
      for (let s = 0; s < o.length; s += 1)
        o[s] && o[s].m(l, i);
      I(l, e, i);
    },
    p(l, i) {
      if (i[0] & /*progress*/
      128) {
        t = Xt(
          /*progress*/
          l[7]
        );
        let s;
        for (s = 0; s < t.length; s += 1) {
          const r = Yo(l, t, s);
          o[s] ? o[s].p(r, i) : (o[s] = xo(r), o[s].c(), o[s].m(e.parentNode, e));
        }
        for (; s < o.length; s += 1)
          o[s].d(1);
        o.length = t.length;
      }
    },
    d(l) {
      l && N(e), Il(o, l);
    }
  };
}
function Qo(n) {
  let e, t = (
    /*p*/
    n[39].unit + ""
  ), o, l, i = " ", s;
  function r(u, c) {
    return (
      /*p*/
      u[39].length != null ? bu : gu
    );
  }
  let a = r(n), _ = a(n);
  return {
    c() {
      _.c(), e = Ie(), o = H(t), l = H(" | "), s = H(i);
    },
    m(u, c) {
      _.m(u, c), I(u, e, c), I(u, o, c), I(u, l, c), I(u, s, c);
    },
    p(u, c) {
      a === (a = r(u)) && _ ? _.p(u, c) : (_.d(1), _ = a(u), _ && (_.c(), _.m(e.parentNode, e))), c[0] & /*progress*/
      128 && t !== (t = /*p*/
      u[39].unit + "") && ke(o, t);
    },
    d(u) {
      u && (N(e), N(o), N(l), N(s)), _.d(u);
    }
  };
}
function gu(n) {
  let e = it(
    /*p*/
    n[39].index || 0
  ) + "", t;
  return {
    c() {
      t = H(e);
    },
    m(o, l) {
      I(o, t, l);
    },
    p(o, l) {
      l[0] & /*progress*/
      128 && e !== (e = it(
        /*p*/
        o[39].index || 0
      ) + "") && ke(t, e);
    },
    d(o) {
      o && N(t);
    }
  };
}
function bu(n) {
  let e = it(
    /*p*/
    n[39].index || 0
  ) + "", t, o, l = it(
    /*p*/
    n[39].length
  ) + "", i;
  return {
    c() {
      t = H(e), o = H("/"), i = H(l);
    },
    m(s, r) {
      I(s, t, r), I(s, o, r), I(s, i, r);
    },
    p(s, r) {
      r[0] & /*progress*/
      128 && e !== (e = it(
        /*p*/
        s[39].index || 0
      ) + "") && ke(t, e), r[0] & /*progress*/
      128 && l !== (l = it(
        /*p*/
        s[39].length
      ) + "") && ke(i, l);
    },
    d(s) {
      s && (N(t), N(o), N(i));
    }
  };
}
function xo(n) {
  let e, t = (
    /*p*/
    n[39].index != null && Qo(n)
  );
  return {
    c() {
      t && t.c(), e = wt();
    },
    m(o, l) {
      t && t.m(o, l), I(o, e, l);
    },
    p(o, l) {
      /*p*/
      o[39].index != null ? t ? t.p(o, l) : (t = Qo(o), t.c(), t.m(e.parentNode, e)) : t && (t.d(1), t = null);
    },
    d(o) {
      o && N(e), t && t.d(o);
    }
  };
}
function el(n) {
  let e, t = (
    /*eta*/
    n[0] ? `/${/*formatted_eta*/
    n[19]}` : ""
  ), o, l;
  return {
    c() {
      e = H(
        /*formatted_timer*/
        n[20]
      ), o = H(t), l = H("s");
    },
    m(i, s) {
      I(i, e, s), I(i, o, s), I(i, l, s);
    },
    p(i, s) {
      s[0] & /*formatted_timer*/
      1048576 && ke(
        e,
        /*formatted_timer*/
        i[20]
      ), s[0] & /*eta, formatted_eta*/
      524289 && t !== (t = /*eta*/
      i[0] ? `/${/*formatted_eta*/
      i[19]}` : "") && ke(o, t);
    },
    d(i) {
      i && (N(e), N(o), N(l));
    }
  };
}
function wu(n) {
  let e, t;
  return e = new Q_({
    props: { margin: (
      /*variant*/
      n[8] === "default"
    ) }
  }), {
    c() {
      eu(e.$$.fragment);
    },
    m(o, l) {
      su(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*variant*/
      256 && (i.margin = /*variant*/
      o[8] === "default"), e.$set(i);
    },
    i(o) {
      t || (gt(e.$$.fragment, o), t = !0);
    },
    o(o) {
      bt(e.$$.fragment, o), t = !1;
    },
    d(o) {
      nu(e, o);
    }
  };
}
function vu(n) {
  let e, t, o, l, i, s = `${/*last_progress_level*/
  n[15] * 100}%`, r = (
    /*progress*/
    n[7] != null && tl(n)
  );
  return {
    c() {
      e = Le("div"), t = Le("div"), r && r.c(), o = Ie(), l = Le("div"), i = Le("div"), Ne(t, "class", "progress-level-inner svelte-1txqlrd"), Ne(i, "class", "progress-bar svelte-1txqlrd"), Ge(i, "width", s), Ne(l, "class", "progress-bar-wrap svelte-1txqlrd"), Ne(e, "class", "progress-level svelte-1txqlrd");
    },
    m(a, _) {
      I(a, e, _), xe(e, t), r && r.m(t, null), xe(e, o), xe(e, l), xe(l, i), n[30](i);
    },
    p(a, _) {
      /*progress*/
      a[7] != null ? r ? r.p(a, _) : (r = tl(a), r.c(), r.m(t, null)) : r && (r.d(1), r = null), _[0] & /*last_progress_level*/
      32768 && s !== (s = `${/*last_progress_level*/
      a[15] * 100}%`) && Ge(i, "width", s);
    },
    i: Zn,
    o: Zn,
    d(a) {
      a && N(e), r && r.d(), n[30](null);
    }
  };
}
function tl(n) {
  let e, t = Xt(
    /*progress*/
    n[7]
  ), o = [];
  for (let l = 0; l < t.length; l += 1)
    o[l] = sl(Xo(n, t, l));
  return {
    c() {
      for (let l = 0; l < o.length; l += 1)
        o[l].c();
      e = wt();
    },
    m(l, i) {
      for (let s = 0; s < o.length; s += 1)
        o[s] && o[s].m(l, i);
      I(l, e, i);
    },
    p(l, i) {
      if (i[0] & /*progress_level, progress*/
      16512) {
        t = Xt(
          /*progress*/
          l[7]
        );
        let s;
        for (s = 0; s < t.length; s += 1) {
          const r = Xo(l, t, s);
          o[s] ? o[s].p(r, i) : (o[s] = sl(r), o[s].c(), o[s].m(e.parentNode, e));
        }
        for (; s < o.length; s += 1)
          o[s].d(1);
        o.length = t.length;
      }
    },
    d(l) {
      l && N(e), Il(o, l);
    }
  };
}
function nl(n) {
  let e, t, o, l, i = (
    /*i*/
    n[41] !== 0 && $u()
  ), s = (
    /*p*/
    n[39].desc != null && ol(n)
  ), r = (
    /*p*/
    n[39].desc != null && /*progress_level*/
    n[14] && /*progress_level*/
    n[14][
      /*i*/
      n[41]
    ] != null && ll()
  ), a = (
    /*progress_level*/
    n[14] != null && il(n)
  );
  return {
    c() {
      i && i.c(), e = Ie(), s && s.c(), t = Ie(), r && r.c(), o = Ie(), a && a.c(), l = wt();
    },
    m(_, u) {
      i && i.m(_, u), I(_, e, u), s && s.m(_, u), I(_, t, u), r && r.m(_, u), I(_, o, u), a && a.m(_, u), I(_, l, u);
    },
    p(_, u) {
      /*p*/
      _[39].desc != null ? s ? s.p(_, u) : (s = ol(_), s.c(), s.m(t.parentNode, t)) : s && (s.d(1), s = null), /*p*/
      _[39].desc != null && /*progress_level*/
      _[14] && /*progress_level*/
      _[14][
        /*i*/
        _[41]
      ] != null ? r || (r = ll(), r.c(), r.m(o.parentNode, o)) : r && (r.d(1), r = null), /*progress_level*/
      _[14] != null ? a ? a.p(_, u) : (a = il(_), a.c(), a.m(l.parentNode, l)) : a && (a.d(1), a = null);
    },
    d(_) {
      _ && (N(e), N(t), N(o), N(l)), i && i.d(_), s && s.d(_), r && r.d(_), a && a.d(_);
    }
  };
}
function $u(n) {
  let e;
  return {
    c() {
      e = H(" /");
    },
    m(t, o) {
      I(t, e, o);
    },
    d(t) {
      t && N(e);
    }
  };
}
function ol(n) {
  let e = (
    /*p*/
    n[39].desc + ""
  ), t;
  return {
    c() {
      t = H(e);
    },
    m(o, l) {
      I(o, t, l);
    },
    p(o, l) {
      l[0] & /*progress*/
      128 && e !== (e = /*p*/
      o[39].desc + "") && ke(t, e);
    },
    d(o) {
      o && N(t);
    }
  };
}
function ll(n) {
  let e;
  return {
    c() {
      e = H("-");
    },
    m(t, o) {
      I(t, e, o);
    },
    d(t) {
      t && N(e);
    }
  };
}
function il(n) {
  let e = (100 * /*progress_level*/
  (n[14][
    /*i*/
    n[41]
  ] || 0)).toFixed(1) + "", t, o;
  return {
    c() {
      t = H(e), o = H("%");
    },
    m(l, i) {
      I(l, t, i), I(l, o, i);
    },
    p(l, i) {
      i[0] & /*progress_level*/
      16384 && e !== (e = (100 * /*progress_level*/
      (l[14][
        /*i*/
        l[41]
      ] || 0)).toFixed(1) + "") && ke(t, e);
    },
    d(l) {
      l && (N(t), N(o));
    }
  };
}
function sl(n) {
  let e, t = (
    /*p*/
    (n[39].desc != null || /*progress_level*/
    n[14] && /*progress_level*/
    n[14][
      /*i*/
      n[41]
    ] != null) && nl(n)
  );
  return {
    c() {
      t && t.c(), e = wt();
    },
    m(o, l) {
      t && t.m(o, l), I(o, e, l);
    },
    p(o, l) {
      /*p*/
      o[39].desc != null || /*progress_level*/
      o[14] && /*progress_level*/
      o[14][
        /*i*/
        o[41]
      ] != null ? t ? t.p(o, l) : (t = nl(o), t.c(), t.m(e.parentNode, e)) : t && (t.d(1), t = null);
    },
    d(o) {
      o && N(e), t && t.d(o);
    }
  };
}
function al(n) {
  let e, t;
  return {
    c() {
      e = Le("p"), t = H(
        /*loading_text*/
        n[9]
      ), Ne(e, "class", "loading svelte-1txqlrd");
    },
    m(o, l) {
      I(o, e, l), xe(e, t);
    },
    p(o, l) {
      l[0] & /*loading_text*/
      512 && ke(
        t,
        /*loading_text*/
        o[9]
      );
    },
    d(o) {
      o && N(e);
    }
  };
}
function ku(n) {
  let e, t, o, l, i;
  const s = [du, cu], r = [];
  function a(_, u) {
    return (
      /*status*/
      _[4] === "pending" ? 0 : (
        /*status*/
        _[4] === "error" ? 1 : -1
      )
    );
  }
  return ~(t = a(n)) && (o = r[t] = s[t](n)), {
    c() {
      e = Le("div"), o && o.c(), Ne(e, "class", l = "wrap " + /*variant*/
      n[8] + " " + /*show_progress*/
      n[6] + " svelte-1txqlrd"), ve(e, "hide", !/*status*/
      n[4] || /*status*/
      n[4] === "complete" || /*show_progress*/
      n[6] === "hidden"), ve(
        e,
        "translucent",
        /*variant*/
        n[8] === "center" && /*status*/
        (n[4] === "pending" || /*status*/
        n[4] === "error") || /*translucent*/
        n[11] || /*show_progress*/
        n[6] === "minimal"
      ), ve(
        e,
        "generating",
        /*status*/
        n[4] === "generating"
      ), ve(
        e,
        "border",
        /*border*/
        n[12]
      ), Ge(
        e,
        "position",
        /*absolute*/
        n[10] ? "absolute" : "static"
      ), Ge(
        e,
        "padding",
        /*absolute*/
        n[10] ? "0" : "var(--size-8) 0"
      );
    },
    m(_, u) {
      I(_, e, u), ~t && r[t].m(e, null), n[31](e), i = !0;
    },
    p(_, u) {
      let c = t;
      t = a(_), t === c ? ~t && r[t].p(_, u) : (o && (Bl(), bt(r[c], 1, 1, () => {
        r[c] = null;
      }), Nl()), ~t ? (o = r[t], o ? o.p(_, u) : (o = r[t] = s[t](_), o.c()), gt(o, 1), o.m(e, null)) : o = null), (!i || u[0] & /*variant, show_progress*/
      320 && l !== (l = "wrap " + /*variant*/
      _[8] + " " + /*show_progress*/
      _[6] + " svelte-1txqlrd")) && Ne(e, "class", l), (!i || u[0] & /*variant, show_progress, status, show_progress*/
      336) && ve(e, "hide", !/*status*/
      _[4] || /*status*/
      _[4] === "complete" || /*show_progress*/
      _[6] === "hidden"), (!i || u[0] & /*variant, show_progress, variant, status, translucent, show_progress*/
      2384) && ve(
        e,
        "translucent",
        /*variant*/
        _[8] === "center" && /*status*/
        (_[4] === "pending" || /*status*/
        _[4] === "error") || /*translucent*/
        _[11] || /*show_progress*/
        _[6] === "minimal"
      ), (!i || u[0] & /*variant, show_progress, status*/
      336) && ve(
        e,
        "generating",
        /*status*/
        _[4] === "generating"
      ), (!i || u[0] & /*variant, show_progress, border*/
      4416) && ve(
        e,
        "border",
        /*border*/
        _[12]
      ), u[0] & /*absolute*/
      1024 && Ge(
        e,
        "position",
        /*absolute*/
        _[10] ? "absolute" : "static"
      ), u[0] & /*absolute*/
      1024 && Ge(
        e,
        "padding",
        /*absolute*/
        _[10] ? "0" : "var(--size-8) 0"
      );
    },
    i(_) {
      i || (gt(o), i = !0);
    },
    o(_) {
      bt(o), i = !1;
    },
    d(_) {
      _ && N(e), ~t && r[t].d(), n[31](null);
    }
  };
}
var yu = function(n, e, t, o) {
  function l(i) {
    return i instanceof t ? i : new t(function(s) {
      s(i);
    });
  }
  return new (t || (t = Promise))(function(i, s) {
    function r(u) {
      try {
        _(o.next(u));
      } catch (c) {
        s(c);
      }
    }
    function a(u) {
      try {
        _(o.throw(u));
      } catch (c) {
        s(c);
      }
    }
    function _(u) {
      u.done ? i(u.value) : l(u.value).then(r, a);
    }
    _((o = o.apply(n, e || [])).next());
  });
};
let Ot = [], Sn = !1;
function qu(n) {
  return yu(this, arguments, void 0, function* (e, t = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && t !== !0)) {
      if (Ot.push(e), !Sn) Sn = !0;
      else return;
      yield _u(), requestAnimationFrame(() => {
        let o = [0, 0];
        for (let l = 0; l < Ot.length; l++) {
          const s = Ot[l].getBoundingClientRect();
          (l === 0 || s.top + window.scrollY <= o[0]) && (o[0] = s.top + window.scrollY, o[1] = l);
        }
        window.scrollTo({ top: o[0] - 20, behavior: "smooth" }), Sn = !1, Ot = [];
      });
    }
  });
}
function Su(n, e, t) {
  let o, { $$slots: l = {}, $$scope: i } = e;
  this && this.__awaiter;
  let { i18n: s } = e, { eta: r = null } = e, { queue: a = !1 } = e, { queue_position: _ } = e, { queue_size: u } = e, { status: c } = e, { scroll_to_output: d = !1 } = e, { timer: f = !0 } = e, { show_progress: p = "full" } = e, { message: g = null } = e, { progress: S = null } = e, { variant: b = "default" } = e, { loading_text: y = "Loading..." } = e, { absolute: h = !0 } = e, { translucent: w = !1 } = e, { border: C = !1 } = e, { autoscroll: T } = e, L, A = !1, ee = 0, j = 0, le = null, re = 0, ne = null, M, F = null, G = !0;
  const Z = () => {
    t(25, ee = performance.now()), t(26, j = 0), A = !0, E();
  };
  function E() {
    requestAnimationFrame(() => {
      t(26, j = (performance.now() - ee) / 1e3), A && E();
    });
  }
  function V() {
    t(26, j = 0), A && (A = !1);
  }
  uu(() => {
    A && V();
  });
  let q = null;
  function k(m) {
    Ho[m ? "unshift" : "push"](() => {
      F = m, t(16, F), t(7, S), t(14, ne), t(15, M);
    });
  }
  function v(m) {
    Ho[m ? "unshift" : "push"](() => {
      L = m, t(13, L);
    });
  }
  return n.$$set = (m) => {
    "i18n" in m && t(1, s = m.i18n), "eta" in m && t(0, r = m.eta), "queue" in m && t(21, a = m.queue), "queue_position" in m && t(2, _ = m.queue_position), "queue_size" in m && t(3, u = m.queue_size), "status" in m && t(4, c = m.status), "scroll_to_output" in m && t(22, d = m.scroll_to_output), "timer" in m && t(5, f = m.timer), "show_progress" in m && t(6, p = m.show_progress), "message" in m && t(23, g = m.message), "progress" in m && t(7, S = m.progress), "variant" in m && t(8, b = m.variant), "loading_text" in m && t(9, y = m.loading_text), "absolute" in m && t(10, h = m.absolute), "translucent" in m && t(11, w = m.translucent), "border" in m && t(12, C = m.border), "autoscroll" in m && t(24, T = m.autoscroll), "$$scope" in m && t(28, i = m.$$scope);
  }, n.$$.update = () => {
    n.$$.dirty[0] & /*eta, old_eta, queue, timer_start*/
    169869313 && (r === null ? t(0, r = le) : a && t(0, r = (performance.now() - ee) / 1e3 + r), r != null && (t(19, q = r.toFixed(1)), t(27, le = r))), n.$$.dirty[0] & /*eta, timer_diff*/
    67108865 && t(17, re = r === null || r <= 0 || !j ? null : Math.min(j / r, 1)), n.$$.dirty[0] & /*progress*/
    128 && S != null && t(18, G = !1), n.$$.dirty[0] & /*progress, progress_level, progress_bar, last_progress_level*/
    114816 && (S != null ? t(14, ne = S.map((m) => {
      if (m.index != null && m.length != null)
        return m.index / m.length;
      if (m.progress != null)
        return m.progress;
    })) : t(14, ne = null), ne ? (t(15, M = ne[ne.length - 1]), F && (M === 0 ? t(16, F.style.transition = "0", F) : t(16, F.style.transition = "150ms", F))) : t(15, M = void 0)), n.$$.dirty[0] & /*status*/
    16 && (c === "pending" ? Z() : V()), n.$$.dirty[0] & /*el, scroll_to_output, status, autoscroll*/
    20979728 && L && d && (c === "pending" || c === "complete") && qu(L, T), n.$$.dirty[0] & /*status, message*/
    8388624, n.$$.dirty[0] & /*timer_diff*/
    67108864 && t(20, o = j.toFixed(1));
  }, [
    r,
    s,
    _,
    u,
    c,
    f,
    p,
    S,
    b,
    y,
    h,
    w,
    C,
    L,
    ne,
    M,
    F,
    re,
    G,
    q,
    o,
    a,
    d,
    g,
    T,
    ee,
    j,
    le,
    i,
    l,
    k,
    v
  ];
}
class Tl extends x_ {
  constructor(e) {
    super(), iu(
      this,
      e,
      Su,
      ku,
      au,
      {
        i18n: 1,
        eta: 0,
        queue: 21,
        queue_position: 2,
        queue_size: 3,
        status: 4,
        scroll_to_output: 22,
        timer: 5,
        show_progress: 6,
        message: 23,
        progress: 7,
        variant: 8,
        loading_text: 9,
        absolute: 10,
        translucent: 11,
        border: 12,
        autoscroll: 24
      },
      null,
      [-1, -1]
    );
  }
}
const {
  SvelteComponent: V2,
  add_render_callback: G2,
  append: H2,
  attr: J2,
  bubble: X2,
  check_outros: Y2,
  create_component: K2,
  create_in_transition: Q2,
  create_out_transition: x2,
  destroy_component: eb,
  detach: tb,
  element: nb,
  group_outros: ob,
  init: lb,
  insert: ib,
  listen: sb,
  mount_component: ab,
  run_all: rb,
  safe_not_equal: _b,
  set_data: ub,
  space: fb,
  stop_propagation: cb,
  text: db,
  transition_in: mb,
  transition_out: pb
} = window.__gradio__svelte__internal, { createEventDispatcher: hb, onMount: gb } = window.__gradio__svelte__internal, {
  SvelteComponent: bb,
  append: wb,
  attr: vb,
  bubble: $b,
  check_outros: kb,
  create_animation: yb,
  create_component: qb,
  destroy_component: Sb,
  detach: Cb,
  element: Eb,
  ensure_array_like: Db,
  fix_and_outro_and_destroy_block: Mb,
  fix_position: zb,
  group_outros: Nb,
  init: Ib,
  insert: Bb,
  mount_component: Tb,
  noop: Lb,
  safe_not_equal: jb,
  set_style: Fb,
  space: Pb,
  transition_in: Ob,
  transition_out: Ab,
  update_keyed_each: Rb
} = window.__gradio__svelte__internal, { setContext: Ub, getContext: Cu } = window.__gradio__svelte__internal, Eu = "WORKER_PROXY_CONTEXT_KEY";
function Du() {
  return Cu(Eu);
}
function Mu(n) {
  return n.host === window.location.host || n.host === "localhost:7860" || n.host === "127.0.0.1:7860" || // Ref: https://github.com/gradio-app/gradio/blob/v3.32.0/js/app/src/Index.svelte#L194
  n.host === "lite.local";
}
async function rl(n) {
  if (n == null)
    return n;
  const e = new URL(n);
  if (!Mu(e) || e.protocol !== "http:" && e.protocol !== "https:")
    return n;
  const t = Du();
  if (t == null)
    return n;
  const o = e.pathname;
  return t.httpRequest({
    method: "GET",
    path: o,
    headers: {},
    query_string: ""
  }).then((l) => {
    if (l.status !== 200)
      throw new Error(`Failed to get file ${o} from the Wasm worker.`);
    const i = new Blob([l.body], {
      type: l.headers["Content-Type"]
    });
    return URL.createObjectURL(i);
  });
}
const {
  SvelteComponent: zu,
  append: Nu,
  assign: Wn,
  compute_rest_props: _l,
  detach: Vn,
  element: Ll,
  empty: Iu,
  exclude_internal_props: Bu,
  get_spread_update: Tu,
  handle_promise: ul,
  init: Lu,
  insert: Gn,
  noop: st,
  safe_not_equal: ju,
  set_attributes: fl,
  set_data: Fu,
  set_style: Pu,
  src_url_equal: Ou,
  text: Au,
  update_await_block_branch: Ru
} = window.__gradio__svelte__internal;
function Uu(n) {
  let e, t = (
    /*error*/
    n[3].message + ""
  ), o;
  return {
    c() {
      e = Ll("p"), o = Au(t), Pu(e, "color", "red");
    },
    m(l, i) {
      Gn(l, e, i), Nu(e, o);
    },
    p(l, i) {
      i & /*src*/
      1 && t !== (t = /*error*/
      l[3].message + "") && Fu(o, t);
    },
    d(l) {
      l && Vn(e);
    }
  };
}
function Zu(n) {
  let e, t, o = [
    {
      src: t = /*resolved_src*/
      n[2]
    },
    /*$$restProps*/
    n[1]
  ], l = {};
  for (let i = 0; i < o.length; i += 1)
    l = Wn(l, o[i]);
  return {
    c() {
      e = Ll("img"), fl(e, l);
    },
    m(i, s) {
      Gn(i, e, s);
    },
    p(i, s) {
      fl(e, l = Tu(o, [
        s & /*src*/
        1 && !Ou(e.src, t = /*resolved_src*/
        i[2]) && { src: t },
        s & /*$$restProps*/
        2 && /*$$restProps*/
        i[1]
      ]));
    },
    d(i) {
      i && Vn(e);
    }
  };
}
function Wu(n) {
  return { c: st, m: st, p: st, d: st };
}
function Vu(n) {
  let e, t, o = {
    ctx: n,
    current: null,
    token: null,
    hasCatch: !0,
    pending: Wu,
    then: Zu,
    catch: Uu,
    value: 2,
    error: 3
  };
  return ul(t = rl(
    /*src*/
    n[0]
  ), o), {
    c() {
      e = Iu(), o.block.c();
    },
    m(l, i) {
      Gn(l, e, i), o.block.m(l, o.anchor = i), o.mount = () => e.parentNode, o.anchor = e;
    },
    p(l, [i]) {
      n = l, o.ctx = n, i & /*src*/
      1 && t !== (t = rl(
        /*src*/
        n[0]
      )) && ul(t, o) || Ru(o, n, i);
    },
    i: st,
    o: st,
    d(l) {
      l && Vn(e), o.block.d(l), o.token = null, o = null;
    }
  };
}
function Gu(n, e, t) {
  const o = ["src"];
  let l = _l(e, o), { src: i = void 0 } = e;
  return n.$$set = (s) => {
    e = Wn(Wn({}, e), Bu(s)), t(1, l = _l(e, o)), "src" in s && t(0, i = s.src);
  }, [i, l];
}
class Hu extends zu {
  constructor(e) {
    super(), Lu(this, e, Gu, Vu, ju, { src: 0 });
  }
}
const {
  SvelteComponent: Ju,
  attr: Xu,
  create_component: Yu,
  destroy_component: Ku,
  detach: Qu,
  element: xu,
  init: ef,
  insert: tf,
  mount_component: nf,
  safe_not_equal: of,
  toggle_class: ot,
  transition_in: lf,
  transition_out: sf
} = window.__gradio__svelte__internal;
function af(n) {
  let e, t, o;
  return t = new Hu({
    props: {
      src: (
        /*samples_dir*/
        n[1] + /*value*/
        n[0]
      ),
      alt: ""
    }
  }), {
    c() {
      e = xu("div"), Yu(t.$$.fragment), Xu(e, "class", "container svelte-h11ksk"), ot(
        e,
        "table",
        /*type*/
        n[2] === "table"
      ), ot(
        e,
        "gallery",
        /*type*/
        n[2] === "gallery"
      ), ot(
        e,
        "selected",
        /*selected*/
        n[3]
      );
    },
    m(l, i) {
      tf(l, e, i), nf(t, e, null), o = !0;
    },
    p(l, [i]) {
      const s = {};
      i & /*samples_dir, value*/
      3 && (s.src = /*samples_dir*/
      l[1] + /*value*/
      l[0]), t.$set(s), (!o || i & /*type*/
      4) && ot(
        e,
        "table",
        /*type*/
        l[2] === "table"
      ), (!o || i & /*type*/
      4) && ot(
        e,
        "gallery",
        /*type*/
        l[2] === "gallery"
      ), (!o || i & /*selected*/
      8) && ot(
        e,
        "selected",
        /*selected*/
        l[3]
      );
    },
    i(l) {
      o || (lf(t.$$.fragment, l), o = !0);
    },
    o(l) {
      sf(t.$$.fragment, l), o = !1;
    },
    d(l) {
      l && Qu(e), Ku(t);
    }
  };
}
function rf(n, e, t) {
  let { value: o } = e, { samples_dir: l } = e, { type: i } = e, { selected: s = !1 } = e;
  return n.$$set = (r) => {
    "value" in r && t(0, o = r.value), "samples_dir" in r && t(1, l = r.samples_dir), "type" in r && t(2, i = r.type), "selected" in r && t(3, s = r.selected);
  }, [o, l, i, s];
}
class Zb extends Ju {
  constructor(e) {
    super(), ef(this, e, rf, af, of, {
      value: 0,
      samples_dir: 1,
      type: 2,
      selected: 3
    });
  }
}
const {
  SvelteComponent: _f,
  add_flush_callback: Cn,
  assign: jl,
  bind: En,
  binding_callbacks: Dn,
  bubble: uf,
  check_outros: Fl,
  create_component: je,
  destroy_component: Fe,
  detach: Kt,
  empty: Pl,
  flush: Q,
  get_spread_object: Ol,
  get_spread_update: Al,
  group_outros: Rl,
  init: ff,
  insert: Qt,
  mount_component: Pe,
  safe_not_equal: cf,
  space: Ul,
  transition_in: ce,
  transition_out: de
} = window.__gradio__svelte__internal;
function df(n) {
  let e, t;
  return e = new ml({
    props: {
      visible: (
        /*visible*/
        n[4]
      ),
      variant: (
        /*_image*/
        n[20] === null ? "dashed" : "solid"
      ),
      border_mode: (
        /*dragging*/
        n[21] ? "focus" : "base"
      ),
      padding: !1,
      elem_id: (
        /*elem_id*/
        n[2]
      ),
      elem_classes: (
        /*elem_classes*/
        n[3]
      ),
      height: (
        /*height*/
        n[9] || void 0
      ),
      width: (
        /*width*/
        n[10]
      ),
      allow_overflow: !1,
      container: (
        /*container*/
        n[12]
      ),
      scale: (
        /*scale*/
        n[13]
      ),
      min_width: (
        /*min_width*/
        n[14]
      ),
      $$slots: { default: [wf] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      je(e.$$.fragment);
    },
    m(o, l) {
      Pe(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*visible*/
      16 && (i.visible = /*visible*/
      o[4]), l[0] & /*_image*/
      1048576 && (i.variant = /*_image*/
      o[20] === null ? "dashed" : "solid"), l[0] & /*dragging*/
      2097152 && (i.border_mode = /*dragging*/
      o[21] ? "focus" : "base"), l[0] & /*elem_id*/
      4 && (i.elem_id = /*elem_id*/
      o[2]), l[0] & /*elem_classes*/
      8 && (i.elem_classes = /*elem_classes*/
      o[3]), l[0] & /*height*/
      512 && (i.height = /*height*/
      o[9] || void 0), l[0] & /*width*/
      1024 && (i.width = /*width*/
      o[10]), l[0] & /*container*/
      4096 && (i.container = /*container*/
      o[12]), l[0] & /*scale*/
      8192 && (i.scale = /*scale*/
      o[13]), l[0] & /*min_width*/
      16384 && (i.min_width = /*min_width*/
      o[14]), l[0] & /*root, sources, label, show_label, streaming, gradio, active_tool, _image, _points, value, dragging, loading_status*/
      16580963 | l[1] & /*$$scope*/
      4096 && (i.$$scope = { dirty: l, ctx: o }), e.$set(i);
    },
    i(o) {
      t || (ce(e.$$.fragment, o), t = !0);
    },
    o(o) {
      de(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Fe(e, o);
    }
  };
}
function mf(n) {
  let e, t;
  return e = new ml({
    props: {
      visible: (
        /*visible*/
        n[4]
      ),
      variant: "solid",
      border_mode: (
        /*dragging*/
        n[21] ? "focus" : "base"
      ),
      padding: !1,
      elem_id: (
        /*elem_id*/
        n[2]
      ),
      elem_classes: (
        /*elem_classes*/
        n[3]
      ),
      height: (
        /*height*/
        n[9] || void 0
      ),
      width: (
        /*width*/
        n[10]
      ),
      allow_overflow: !1,
      container: (
        /*container*/
        n[12]
      ),
      scale: (
        /*scale*/
        n[13]
      ),
      min_width: (
        /*min_width*/
        n[14]
      ),
      $$slots: { default: [vf] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      je(e.$$.fragment);
    },
    m(o, l) {
      Pe(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*visible*/
      16 && (i.visible = /*visible*/
      o[4]), l[0] & /*dragging*/
      2097152 && (i.border_mode = /*dragging*/
      o[21] ? "focus" : "base"), l[0] & /*elem_id*/
      4 && (i.elem_id = /*elem_id*/
      o[2]), l[0] & /*elem_classes*/
      8 && (i.elem_classes = /*elem_classes*/
      o[3]), l[0] & /*height*/
      512 && (i.height = /*height*/
      o[9] || void 0), l[0] & /*width*/
      1024 && (i.width = /*width*/
      o[10]), l[0] & /*container*/
      4096 && (i.container = /*container*/
      o[12]), l[0] & /*scale*/
      8192 && (i.scale = /*scale*/
      o[13]), l[0] & /*min_width*/
      16384 && (i.min_width = /*min_width*/
      o[14]), l[0] & /*_image, label, show_label, show_download_button, _selectable, show_share_button, gradio, loading_status*/
      1607906 | l[1] & /*$$scope*/
      4096 && (i.$$scope = { dirty: l, ctx: o }), e.$set(i);
    },
    i(o) {
      t || (ce(e.$$.fragment, o), t = !0);
    },
    o(o) {
      de(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Fe(e, o);
    }
  };
}
function pf(n) {
  let e, t;
  return e = new bl({
    props: {
      unpadded_box: !0,
      size: "large",
      $$slots: { default: [gf] },
      $$scope: { ctx: n }
    }
  }), {
    c() {
      je(e.$$.fragment);
    },
    m(o, l) {
      Pe(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[1] & /*$$scope*/
      4096 && (i.$$scope = { dirty: l, ctx: o }), e.$set(i);
    },
    i(o) {
      t || (ce(e.$$.fragment, o), t = !0);
    },
    o(o) {
      de(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Fe(e, o);
    }
  };
}
function hf(n) {
  let e, t;
  return e = new qa({
    props: {
      i18n: (
        /*gradio*/
        n[19].i18n
      ),
      type: "image",
      mode: "short"
    }
  }), {
    c() {
      je(e.$$.fragment);
    },
    m(o, l) {
      Pe(e, o, l), t = !0;
    },
    p(o, l) {
      const i = {};
      l[0] & /*gradio*/
      524288 && (i.i18n = /*gradio*/
      o[19].i18n), e.$set(i);
    },
    i(o) {
      t || (ce(e.$$.fragment, o), t = !0);
    },
    o(o) {
      de(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Fe(e, o);
    }
  };
}
function gf(n) {
  let e, t;
  return e = new Yt({}), {
    c() {
      je(e.$$.fragment);
    },
    m(o, l) {
      Pe(e, o, l), t = !0;
    },
    i(o) {
      t || (ce(e.$$.fragment, o), t = !0);
    },
    o(o) {
      de(e.$$.fragment, o), t = !1;
    },
    d(o) {
      Fe(e, o);
    }
  };
}
function bf(n) {
  let e, t, o, l, i;
  const s = [hf, pf], r = [];
  function a(_, u) {
    return u[0] & /*sources*/
    65536 && (e = null), e == null && (e = !!/*sources*/
    _[16].includes("upload")), e ? 0 : 1;
  }
  return t = a(n, [-1, -1]), o = r[t] = s[t](n), {
    c() {
      o.c(), l = Pl();
    },
    m(_, u) {
      r[t].m(_, u), Qt(_, l, u), i = !0;
    },
    p(_, u) {
      let c = t;
      t = a(_, u), t === c ? r[t].p(_, u) : (Rl(), de(r[c], 1, 1, () => {
        r[c] = null;
      }), Fl(), o = r[t], o ? o.p(_, u) : (o = r[t] = s[t](_), o.c()), ce(o, 1), o.m(l.parentNode, l));
    },
    i(_) {
      i || (ce(o), i = !0);
    },
    o(_) {
      de(o), i = !1;
    },
    d(_) {
      _ && Kt(l), r[t].d(_);
    }
  };
}
function wf(n) {
  let e, t, o, l, i, s, r;
  const a = [
    {
      autoscroll: (
        /*gradio*/
        n[19].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      n[19].i18n
    ) },
    /*loading_status*/
    n[1]
  ];
  let _ = {};
  for (let p = 0; p < a.length; p += 1)
    _ = jl(_, a[p]);
  e = new Tl({ props: _ });
  function u(p) {
    n[29](p);
  }
  function c(p) {
    n[30](p);
  }
  function d(p) {
    n[31](p);
  }
  let f = {
    root: (
      /*root*/
      n[8]
    ),
    sources: (
      /*sources*/
      n[16]
    ),
    label: (
      /*label*/
      n[5]
    ),
    show_label: (
      /*show_label*/
      n[6]
    ),
    streaming: (
      /*streaming*/
      n[18]
    ),
    i18n: (
      /*gradio*/
      n[19].i18n
    ),
    $$slots: { default: [bf] },
    $$scope: { ctx: n }
  };
  return (
    /*active_tool*/
    n[22] !== void 0 && (f.active_tool = /*active_tool*/
    n[22]), /*_image*/
    n[20] !== void 0 && (f.value = /*_image*/
    n[20]), /*_points*/
    n[23] !== void 0 && (f.points = /*_points*/
    n[23]), o = new O_({ props: f }), Dn.push(() => En(o, "active_tool", u)), Dn.push(() => En(o, "value", c)), Dn.push(() => En(o, "points", d)), o.$on(
      "points_change",
      /*points_change_handler*/
      n[32]
    ), o.$on(
      "edit",
      /*edit_handler*/
      n[33]
    ), o.$on(
      "clear",
      /*clear_handler*/
      n[34]
    ), o.$on(
      "stream",
      /*stream_handler*/
      n[35]
    ), o.$on(
      "drag",
      /*drag_handler*/
      n[36]
    ), o.$on(
      "upload",
      /*upload_handler*/
      n[37]
    ), o.$on(
      "select",
      /*select_handler_1*/
      n[38]
    ), o.$on(
      "share",
      /*share_handler_1*/
      n[39]
    ), o.$on(
      "error",
      /*error_handler_2*/
      n[40]
    ), o.$on(
      "click",
      /*click_handler*/
      n[41]
    ), o.$on(
      "error",
      /*error_handler*/
      n[42]
    ), {
      c() {
        je(e.$$.fragment), t = Ul(), je(o.$$.fragment);
      },
      m(p, g) {
        Pe(e, p, g), Qt(p, t, g), Pe(o, p, g), r = !0;
      },
      p(p, g) {
        const S = g[0] & /*gradio, loading_status*/
        524290 ? Al(a, [
          g[0] & /*gradio*/
          524288 && {
            autoscroll: (
              /*gradio*/
              p[19].autoscroll
            )
          },
          g[0] & /*gradio*/
          524288 && { i18n: (
            /*gradio*/
            p[19].i18n
          ) },
          g[0] & /*loading_status*/
          2 && Ol(
            /*loading_status*/
            p[1]
          )
        ]) : {};
        e.$set(S);
        const b = {};
        g[0] & /*root*/
        256 && (b.root = /*root*/
        p[8]), g[0] & /*sources*/
        65536 && (b.sources = /*sources*/
        p[16]), g[0] & /*label*/
        32 && (b.label = /*label*/
        p[5]), g[0] & /*show_label*/
        64 && (b.show_label = /*show_label*/
        p[6]), g[0] & /*streaming*/
        262144 && (b.streaming = /*streaming*/
        p[18]), g[0] & /*gradio*/
        524288 && (b.i18n = /*gradio*/
        p[19].i18n), g[0] & /*gradio, sources*/
        589824 | g[1] & /*$$scope*/
        4096 && (b.$$scope = { dirty: g, ctx: p }), !l && g[0] & /*active_tool*/
        4194304 && (l = !0, b.active_tool = /*active_tool*/
        p[22], Cn(() => l = !1)), !i && g[0] & /*_image*/
        1048576 && (i = !0, b.value = /*_image*/
        p[20], Cn(() => i = !1)), !s && g[0] & /*_points*/
        8388608 && (s = !0, b.points = /*_points*/
        p[23], Cn(() => s = !1)), o.$set(b);
      },
      i(p) {
        r || (ce(e.$$.fragment, p), ce(o.$$.fragment, p), r = !0);
      },
      o(p) {
        de(e.$$.fragment, p), de(o.$$.fragment, p), r = !1;
      },
      d(p) {
        p && Kt(t), Fe(e, p), Fe(o, p);
      }
    }
  );
}
function vf(n) {
  let e, t, o, l;
  const i = [
    {
      autoscroll: (
        /*gradio*/
        n[19].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      n[19].i18n
    ) },
    /*loading_status*/
    n[1]
  ];
  let s = {};
  for (let r = 0; r < i.length; r += 1)
    s = jl(s, i[r]);
  return e = new Tl({ props: s }), o = new Qa({
    props: {
      value: (
        /*_image*/
        n[20]
      ),
      label: (
        /*label*/
        n[5]
      ),
      show_label: (
        /*show_label*/
        n[6]
      ),
      show_download_button: (
        /*show_download_button*/
        n[7]
      ),
      selectable: (
        /*_selectable*/
        n[11]
      ),
      show_share_button: (
        /*show_share_button*/
        n[15]
      ),
      i18n: (
        /*gradio*/
        n[19].i18n
      )
    }
  }), o.$on(
    "select",
    /*select_handler*/
    n[26]
  ), o.$on(
    "share",
    /*share_handler*/
    n[27]
  ), o.$on(
    "error",
    /*error_handler_1*/
    n[28]
  ), {
    c() {
      je(e.$$.fragment), t = Ul(), je(o.$$.fragment);
    },
    m(r, a) {
      Pe(e, r, a), Qt(r, t, a), Pe(o, r, a), l = !0;
    },
    p(r, a) {
      const _ = a[0] & /*gradio, loading_status*/
      524290 ? Al(i, [
        a[0] & /*gradio*/
        524288 && {
          autoscroll: (
            /*gradio*/
            r[19].autoscroll
          )
        },
        a[0] & /*gradio*/
        524288 && { i18n: (
          /*gradio*/
          r[19].i18n
        ) },
        a[0] & /*loading_status*/
        2 && Ol(
          /*loading_status*/
          r[1]
        )
      ]) : {};
      e.$set(_);
      const u = {};
      a[0] & /*_image*/
      1048576 && (u.value = /*_image*/
      r[20]), a[0] & /*label*/
      32 && (u.label = /*label*/
      r[5]), a[0] & /*show_label*/
      64 && (u.show_label = /*show_label*/
      r[6]), a[0] & /*show_download_button*/
      128 && (u.show_download_button = /*show_download_button*/
      r[7]), a[0] & /*_selectable*/
      2048 && (u.selectable = /*_selectable*/
      r[11]), a[0] & /*show_share_button*/
      32768 && (u.show_share_button = /*show_share_button*/
      r[15]), a[0] & /*gradio*/
      524288 && (u.i18n = /*gradio*/
      r[19].i18n), o.$set(u);
    },
    i(r) {
      l || (ce(e.$$.fragment, r), ce(o.$$.fragment, r), l = !0);
    },
    o(r) {
      de(e.$$.fragment, r), de(o.$$.fragment, r), l = !1;
    },
    d(r) {
      r && Kt(t), Fe(e, r), Fe(o, r);
    }
  };
}
function $f(n) {
  let e, t, o, l;
  const i = [mf, df], s = [];
  function r(a, _) {
    return (
      /*interactive*/
      a[17] ? 1 : 0
    );
  }
  return e = r(n), t = s[e] = i[e](n), {
    c() {
      t.c(), o = Pl();
    },
    m(a, _) {
      s[e].m(a, _), Qt(a, o, _), l = !0;
    },
    p(a, _) {
      let u = e;
      e = r(a), e === u ? s[e].p(a, _) : (Rl(), de(s[u], 1, 1, () => {
        s[u] = null;
      }), Fl(), t = s[e], t ? t.p(a, _) : (t = s[e] = i[e](a), t.c()), ce(t, 1), t.m(o.parentNode, o));
    },
    i(a) {
      l || (ce(t), l = !0);
    },
    o(a) {
      de(t), l = !1;
    },
    d(a) {
      a && Kt(o), s[e].d(a);
    }
  };
}
function kf(n, e, t) {
  let o, l, i, { elem_id: s = "" } = e, { elem_classes: r = [] } = e, { visible: a = !0 } = e, { value: _ = null } = e, { label: u } = e, { show_label: c } = e, { show_download_button: d } = e, { root: f } = e, { proxy_url: p } = e, { height: g } = e, { width: S } = e, { _selectable: b = !1 } = e, { container: y = !0 } = e, { scale: h = null } = e, { min_width: w = void 0 } = e, { loading_status: C } = e, { show_share_button: T = !1 } = e, { sources: L = ["upload"] } = e, { interactive: A } = e, { streaming: ee } = e, { gradio: j } = e, le, re = null;
  const ne = ({ detail: $ }) => j.dispatch("select", $), M = ({ detail: $ }) => j.dispatch("share", $), F = ({ detail: $ }) => j.dispatch("error", $);
  function G($) {
    re = $, t(22, re);
  }
  function Z($) {
    o = $, t(20, o), t(0, _), t(8, f), t(24, p);
  }
  function E($) {
    l = $, t(23, l), t(0, _);
  }
  const V = ({ detail: $ }) => t(0, _.points = $, _), q = () => j.dispatch("edit"), k = () => {
    t(0, _ = null), j.dispatch("clear"), j.dispatch("change");
  }, v = () => j.dispatch("stream"), m = ({ detail: $ }) => t(21, le = $), D = ({ detail: $ }) => {
    _ == null ? t(0, _ = { image: $, points: null }) : t(0, _.image = $, _), j.dispatch("upload");
  }, z = ({ detail: $ }) => j.dispatch("select", $), P = ({ detail: $ }) => j.dispatch("share", $), W = ({ detail: $ }) => {
    t(1, C), t(1, C.status = "error", C), j.dispatch("error", $);
  }, R = () => j.dispatch("error", "bad thing happened");
  function B($) {
    uf.call(this, n, $);
  }
  return n.$$set = ($) => {
    "elem_id" in $ && t(2, s = $.elem_id), "elem_classes" in $ && t(3, r = $.elem_classes), "visible" in $ && t(4, a = $.visible), "value" in $ && t(0, _ = $.value), "label" in $ && t(5, u = $.label), "show_label" in $ && t(6, c = $.show_label), "show_download_button" in $ && t(7, d = $.show_download_button), "root" in $ && t(8, f = $.root), "proxy_url" in $ && t(24, p = $.proxy_url), "height" in $ && t(9, g = $.height), "width" in $ && t(10, S = $.width), "_selectable" in $ && t(11, b = $._selectable), "container" in $ && t(12, y = $.container), "scale" in $ && t(13, h = $.scale), "min_width" in $ && t(14, w = $.min_width), "loading_status" in $ && t(1, C = $.loading_status), "show_share_button" in $ && t(15, T = $.show_share_button), "sources" in $ && t(16, L = $.sources), "interactive" in $ && t(17, A = $.interactive), "streaming" in $ && t(18, ee = $.streaming), "gradio" in $ && t(19, j = $.gradio);
  }, n.$$.update = () => {
    n.$$.dirty[0] & /*value, root, proxy_url*/
    16777473 && t(20, o = _ && Te(_.image, f, p)), n.$$.dirty[0] & /*value*/
    1 && t(23, l = _ && _.points), n.$$.dirty[0] & /*_image*/
    1048576 && t(25, i = o?.url), n.$$.dirty[0] & /*url, gradio*/
    34078720 && i && j.dispatch("change");
  }, [
    _,
    C,
    s,
    r,
    a,
    u,
    c,
    d,
    f,
    g,
    S,
    b,
    y,
    h,
    w,
    T,
    L,
    A,
    ee,
    j,
    o,
    le,
    re,
    l,
    p,
    i,
    ne,
    M,
    F,
    G,
    Z,
    E,
    V,
    q,
    k,
    v,
    m,
    D,
    z,
    P,
    W,
    R,
    B
  ];
}
class Wb extends _f {
  constructor(e) {
    super(), ff(
      this,
      e,
      kf,
      $f,
      cf,
      {
        elem_id: 2,
        elem_classes: 3,
        visible: 4,
        value: 0,
        label: 5,
        show_label: 6,
        show_download_button: 7,
        root: 8,
        proxy_url: 24,
        height: 9,
        width: 10,
        _selectable: 11,
        container: 12,
        scale: 13,
        min_width: 14,
        loading_status: 1,
        show_share_button: 15,
        sources: 16,
        interactive: 17,
        streaming: 18,
        gradio: 19
      },
      null,
      [-1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[2];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), Q();
  }
  get elem_classes() {
    return this.$$.ctx[3];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), Q();
  }
  get visible() {
    return this.$$.ctx[4];
  }
  set visible(e) {
    this.$$set({ visible: e }), Q();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(e) {
    this.$$set({ value: e }), Q();
  }
  get label() {
    return this.$$.ctx[5];
  }
  set label(e) {
    this.$$set({ label: e }), Q();
  }
  get show_label() {
    return this.$$.ctx[6];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), Q();
  }
  get show_download_button() {
    return this.$$.ctx[7];
  }
  set show_download_button(e) {
    this.$$set({ show_download_button: e }), Q();
  }
  get root() {
    return this.$$.ctx[8];
  }
  set root(e) {
    this.$$set({ root: e }), Q();
  }
  get proxy_url() {
    return this.$$.ctx[24];
  }
  set proxy_url(e) {
    this.$$set({ proxy_url: e }), Q();
  }
  get height() {
    return this.$$.ctx[9];
  }
  set height(e) {
    this.$$set({ height: e }), Q();
  }
  get width() {
    return this.$$.ctx[10];
  }
  set width(e) {
    this.$$set({ width: e }), Q();
  }
  get _selectable() {
    return this.$$.ctx[11];
  }
  set _selectable(e) {
    this.$$set({ _selectable: e }), Q();
  }
  get container() {
    return this.$$.ctx[12];
  }
  set container(e) {
    this.$$set({ container: e }), Q();
  }
  get scale() {
    return this.$$.ctx[13];
  }
  set scale(e) {
    this.$$set({ scale: e }), Q();
  }
  get min_width() {
    return this.$$.ctx[14];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), Q();
  }
  get loading_status() {
    return this.$$.ctx[1];
  }
  set loading_status(e) {
    this.$$set({ loading_status: e }), Q();
  }
  get show_share_button() {
    return this.$$.ctx[15];
  }
  set show_share_button(e) {
    this.$$set({ show_share_button: e }), Q();
  }
  get sources() {
    return this.$$.ctx[16];
  }
  set sources(e) {
    this.$$set({ sources: e }), Q();
  }
  get interactive() {
    return this.$$.ctx[17];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), Q();
  }
  get streaming() {
    return this.$$.ctx[18];
  }
  set streaming(e) {
    this.$$set({ streaming: e }), Q();
  }
  get gradio() {
    return this.$$.ctx[19];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), Q();
  }
}
export {
  Zb as BaseExample,
  Hu as BaseImage,
  O_ as BaseImageUploader,
  Qa as BaseStaticImage,
  k_ as BoxDrawer,
  Wb as default
};
