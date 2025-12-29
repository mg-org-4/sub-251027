import { app as fs } from "../../../scripts/app.js";
/**
* @vue/shared v3.5.18
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
/*! #__NO_SIDE_EFFECTS__ */
// @__NO_SIDE_EFFECTS__
function ki(e) {
  const t = /* @__PURE__ */ Object.create(null);
  for (const n of e.split(",")) t[n] = 1;
  return (n) => n in t;
}
const me = {}, Fn = [], Dt = () => {
}, Vc = () => !1, vo = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && // uppercase letter
(e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), $i = (e) => e.startsWith("onUpdate:"), Ie = Object.assign, Ni = (e, t) => {
  const n = e.indexOf(t);
  n > -1 && e.splice(n, 1);
}, Hc = Object.prototype.hasOwnProperty, ce = (e, t) => Hc.call(e, t), X = Array.isArray, jn = (e) => yo(e) === "[object Map]", pl = (e) => yo(e) === "[object Set]", q = (e) => typeof e == "function", Oe = (e) => typeof e == "string", qt = (e) => typeof e == "symbol", Ee = (e) => e !== null && typeof e == "object", ml = (e) => (Ee(e) || q(e)) && q(e.then) && q(e.catch), hl = Object.prototype.toString, yo = (e) => hl.call(e), Wc = (e) => yo(e).slice(8, -1), gl = (e) => yo(e) === "[object Object]", Ii = (e) => Oe(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, rr = /* @__PURE__ */ ki(
  // the leading comma is intentional so empty string "" is also included
  ",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"
), _o = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => t[n] || (t[n] = e(n));
}, Bc = /-(\w)/g, pt = _o(
  (e) => e.replace(Bc, (t, n) => n ? n.toUpperCase() : "")
), Kc = /\B([A-Z])/g, wn = _o(
  (e) => e.replace(Kc, "-$1").toLowerCase()
), So = _o((e) => e.charAt(0).toUpperCase() + e.slice(1)), Ro = _o(
  (e) => e ? `on${So(e)}` : ""
), ln = (e, t) => !Object.is(e, t), Mo = (e, ...t) => {
  for (let n = 0; n < e.length; n++)
    e[n](...t);
}, ti = (e, t, n, r = !1) => {
  Object.defineProperty(e, t, {
    configurable: !0,
    enumerable: !1,
    writable: r,
    value: n
  });
}, Yc = (e) => {
  const t = parseFloat(e);
  return isNaN(t) ? e : t;
}, zc = (e) => {
  const t = Oe(e) ? Number(e) : NaN;
  return isNaN(t) ? e : t;
};
let ps;
const Eo = () => ps || (ps = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {});
function Ai(e) {
  if (X(e)) {
    const t = {};
    for (let n = 0; n < e.length; n++) {
      const r = e[n], o = Oe(r) ? qc(r) : Ai(r);
      if (o)
        for (const i in o)
          t[i] = o[i];
    }
    return t;
  } else if (Oe(e) || Ee(e))
    return e;
}
const Gc = /;(?![^(]*\))/g, Xc = /:([^]+)/, Jc = /\/\*[^]*?\*\//g;
function qc(e) {
  const t = {};
  return e.replace(Jc, "").split(Gc).forEach((n) => {
    if (n) {
      const r = n.split(Xc);
      r.length > 1 && (t[r[0].trim()] = r[1].trim());
    }
  }), t;
}
function Bn(e) {
  let t = "";
  if (Oe(e))
    t = e;
  else if (X(e))
    for (let n = 0; n < e.length; n++) {
      const r = Bn(e[n]);
      r && (t += r + " ");
    }
  else if (Ee(e))
    for (const n in e)
      e[n] && (t += n + " ");
  return t.trim();
}
const Zc = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", Qc = /* @__PURE__ */ ki(Zc);
function bl(e) {
  return !!e || e === "";
}
const vl = (e) => !!(e && e.__v_isRef === !0), mr = (e) => Oe(e) ? e : e == null ? "" : X(e) || Ee(e) && (e.toString === hl || !q(e.toString)) ? vl(e) ? mr(e.value) : JSON.stringify(e, yl, 2) : String(e), yl = (e, t) => vl(t) ? yl(e, t.value) : jn(t) ? {
  [`Map(${t.size})`]: [...t.entries()].reduce(
    (n, [r, o], i) => (n[Fo(r, i) + " =>"] = o, n),
    {}
  )
} : pl(t) ? {
  [`Set(${t.size})`]: [...t.values()].map((n) => Fo(n))
} : qt(t) ? Fo(t) : Ee(t) && !X(t) && !gl(t) ? String(t) : t, Fo = (e, t = "") => {
  var n;
  return (
    // Symbol.description in es2019+ so we need to cast here to pass
    // the lib: es2016 check
    qt(e) ? `Symbol(${(n = e.description) != null ? n : t})` : e
  );
};
/**
* @vue/reactivity v3.5.18
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
let rt;
class _l {
  constructor(t = !1) {
    this.detached = t, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this.parent = rt, !t && rt && (this.index = (rt.scopes || (rt.scopes = [])).push(
      this
    ) - 1);
  }
  get active() {
    return this._active;
  }
  pause() {
    if (this._active) {
      this._isPaused = !0;
      let t, n;
      if (this.scopes)
        for (t = 0, n = this.scopes.length; t < n; t++)
          this.scopes[t].pause();
      for (t = 0, n = this.effects.length; t < n; t++)
        this.effects[t].pause();
    }
  }
  /**
   * Resumes the effect scope, including all child scopes and effects.
   */
  resume() {
    if (this._active && this._isPaused) {
      this._isPaused = !1;
      let t, n;
      if (this.scopes)
        for (t = 0, n = this.scopes.length; t < n; t++)
          this.scopes[t].resume();
      for (t = 0, n = this.effects.length; t < n; t++)
        this.effects[t].resume();
    }
  }
  run(t) {
    if (this._active) {
      const n = rt;
      try {
        return rt = this, t();
      } finally {
        rt = n;
      }
    }
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  on() {
    ++this._on === 1 && (this.prevScope = rt, rt = this);
  }
  /**
   * This should only be called on non-detached scopes
   * @internal
   */
  off() {
    this._on > 0 && --this._on === 0 && (rt = this.prevScope, this.prevScope = void 0);
  }
  stop(t) {
    if (this._active) {
      this._active = !1;
      let n, r;
      for (n = 0, r = this.effects.length; n < r; n++)
        this.effects[n].stop();
      for (this.effects.length = 0, n = 0, r = this.cleanups.length; n < r; n++)
        this.cleanups[n]();
      if (this.cleanups.length = 0, this.scopes) {
        for (n = 0, r = this.scopes.length; n < r; n++)
          this.scopes[n].stop(!0);
        this.scopes.length = 0;
      }
      if (!this.detached && this.parent && !t) {
        const o = this.parent.scopes.pop();
        o && o !== this && (this.parent.scopes[this.index] = o, o.index = this.index);
      }
      this.parent = void 0;
    }
  }
}
function Sl(e) {
  return new _l(e);
}
function ed() {
  return rt;
}
let ge;
const jo = /* @__PURE__ */ new WeakSet();
class El {
  constructor(t) {
    this.fn = t, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, rt && rt.active && rt.effects.push(this);
  }
  pause() {
    this.flags |= 64;
  }
  resume() {
    this.flags & 64 && (this.flags &= -65, jo.has(this) && (jo.delete(this), this.trigger()));
  }
  /**
   * @internal
   */
  notify() {
    this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Cl(this);
  }
  run() {
    if (!(this.flags & 1))
      return this.fn();
    this.flags |= 2, ms(this), Ol(this);
    const t = ge, n = Et;
    ge = this, Et = !0;
    try {
      return this.fn();
    } finally {
      Ll(this), ge = t, Et = n, this.flags &= -3;
    }
  }
  stop() {
    if (this.flags & 1) {
      for (let t = this.deps; t; t = t.nextDep)
        Ri(t);
      this.deps = this.depsTail = void 0, ms(this), this.onStop && this.onStop(), this.flags &= -2;
    }
  }
  trigger() {
    this.flags & 64 ? jo.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
  }
  /**
   * @internal
   */
  runIfDirty() {
    ni(this) && this.run();
  }
  get dirty() {
    return ni(this);
  }
}
let Tl = 0, or, ir;
function Cl(e, t = !1) {
  if (e.flags |= 8, t) {
    e.next = ir, ir = e;
    return;
  }
  e.next = or, or = e;
}
function xi() {
  Tl++;
}
function Di() {
  if (--Tl > 0)
    return;
  if (ir) {
    let t = ir;
    for (ir = void 0; t; ) {
      const n = t.next;
      t.next = void 0, t.flags &= -9, t = n;
    }
  }
  let e;
  for (; or; ) {
    let t = or;
    for (or = void 0; t; ) {
      const n = t.next;
      if (t.next = void 0, t.flags &= -9, t.flags & 1)
        try {
          t.trigger();
        } catch (r) {
          e || (e = r);
        }
      t = n;
    }
  }
  if (e) throw e;
}
function Ol(e) {
  for (let t = e.deps; t; t = t.nextDep)
    t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Ll(e) {
  let t, n = e.depsTail, r = n;
  for (; r; ) {
    const o = r.prevDep;
    r.version === -1 ? (r === n && (n = o), Ri(r), td(r)) : t = r, r.dep.activeLink = r.prevActiveLink, r.prevActiveLink = void 0, r = o;
  }
  e.deps = t, e.depsTail = n;
}
function ni(e) {
  for (let t = e.deps; t; t = t.nextDep)
    if (t.dep.version !== t.version || t.dep.computed && (Pl(t.dep.computed) || t.dep.version !== t.version))
      return !0;
  return !!e._dirty;
}
function Pl(e) {
  if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === hr) || (e.globalVersion = hr, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !ni(e))))
    return;
  e.flags |= 2;
  const t = e.dep, n = ge, r = Et;
  ge = e, Et = !0;
  try {
    Ol(e);
    const o = e.fn(e._value);
    (t.version === 0 || ln(o, e._value)) && (e.flags |= 128, e._value = o, t.version++);
  } catch (o) {
    throw t.version++, o;
  } finally {
    ge = n, Et = r, Ll(e), e.flags &= -3;
  }
}
function Ri(e, t = !1) {
  const { dep: n, prevSub: r, nextSub: o } = e;
  if (r && (r.nextSub = o, e.prevSub = void 0), o && (o.prevSub = r, e.nextSub = void 0), n.subs === e && (n.subs = r, !r && n.computed)) {
    n.computed.flags &= -5;
    for (let i = n.computed.deps; i; i = i.nextDep)
      Ri(i, !0);
  }
  !t && !--n.sc && n.map && n.map.delete(n.key);
}
function td(e) {
  const { prevDep: t, nextDep: n } = e;
  t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
let Et = !0;
const wl = [];
function Xt() {
  wl.push(Et), Et = !1;
}
function Jt() {
  const e = wl.pop();
  Et = e === void 0 ? !0 : e;
}
function ms(e) {
  const { cleanup: t } = e;
  if (e.cleanup = void 0, t) {
    const n = ge;
    ge = void 0;
    try {
      t();
    } finally {
      ge = n;
    }
  }
}
let hr = 0;
class nd {
  constructor(t, n) {
    this.sub = t, this.dep = n, this.version = n.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
  }
}
class Mi {
  // TODO isolatedDeclarations "__v_skip"
  constructor(t) {
    this.computed = t, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
  }
  track(t) {
    if (!ge || !Et || ge === this.computed)
      return;
    let n = this.activeLink;
    if (n === void 0 || n.sub !== ge)
      n = this.activeLink = new nd(ge, this), ge.deps ? (n.prevDep = ge.depsTail, ge.depsTail.nextDep = n, ge.depsTail = n) : ge.deps = ge.depsTail = n, kl(n);
    else if (n.version === -1 && (n.version = this.version, n.nextDep)) {
      const r = n.nextDep;
      r.prevDep = n.prevDep, n.prevDep && (n.prevDep.nextDep = r), n.prevDep = ge.depsTail, n.nextDep = void 0, ge.depsTail.nextDep = n, ge.depsTail = n, ge.deps === n && (ge.deps = r);
    }
    return n;
  }
  trigger(t) {
    this.version++, hr++, this.notify(t);
  }
  notify(t) {
    xi();
    try {
      for (let n = this.subs; n; n = n.prevSub)
        n.sub.notify() && n.sub.dep.notify();
    } finally {
      Di();
    }
  }
}
function kl(e) {
  if (e.dep.sc++, e.sub.flags & 4) {
    const t = e.dep.computed;
    if (t && !e.dep.subs) {
      t.flags |= 20;
      for (let r = t.deps; r; r = r.nextDep)
        kl(r);
    }
    const n = e.dep.subs;
    n !== e && (e.prevSub = n, n && (n.nextSub = e)), e.dep.subs = e;
  }
}
const ri = /* @__PURE__ */ new WeakMap(), Cn = Symbol(
  ""
), oi = Symbol(
  ""
), gr = Symbol(
  ""
);
function Ue(e, t, n) {
  if (Et && ge) {
    let r = ri.get(e);
    r || ri.set(e, r = /* @__PURE__ */ new Map());
    let o = r.get(n);
    o || (r.set(n, o = new Mi()), o.map = r, o.key = n), o.track();
  }
}
function Yt(e, t, n, r, o, i) {
  const s = ri.get(e);
  if (!s) {
    hr++;
    return;
  }
  const a = (l) => {
    l && l.trigger();
  };
  if (xi(), t === "clear")
    s.forEach(a);
  else {
    const l = X(e), u = l && Ii(n);
    if (l && n === "length") {
      const c = Number(r);
      s.forEach((d, f) => {
        (f === "length" || f === gr || !qt(f) && f >= c) && a(d);
      });
    } else
      switch ((n !== void 0 || s.has(void 0)) && a(s.get(n)), u && a(s.get(gr)), t) {
        case "add":
          l ? u && a(s.get("length")) : (a(s.get(Cn)), jn(e) && a(s.get(oi)));
          break;
        case "delete":
          l || (a(s.get(Cn)), jn(e) && a(s.get(oi)));
          break;
        case "set":
          jn(e) && a(s.get(Cn));
          break;
      }
  }
  Di();
}
function Nn(e) {
  const t = le(e);
  return t === e ? t : (Ue(t, "iterate", gr), Tt(e) ? t : t.map(Je));
}
function Fi(e) {
  return Ue(e = le(e), "iterate", gr), e;
}
const rd = {
  __proto__: null,
  [Symbol.iterator]() {
    return Uo(this, Symbol.iterator, Je);
  },
  concat(...e) {
    return Nn(this).concat(
      ...e.map((t) => X(t) ? Nn(t) : t)
    );
  },
  entries() {
    return Uo(this, "entries", (e) => (e[1] = Je(e[1]), e));
  },
  every(e, t) {
    return Ut(this, "every", e, t, void 0, arguments);
  },
  filter(e, t) {
    return Ut(this, "filter", e, t, (n) => n.map(Je), arguments);
  },
  find(e, t) {
    return Ut(this, "find", e, t, Je, arguments);
  },
  findIndex(e, t) {
    return Ut(this, "findIndex", e, t, void 0, arguments);
  },
  findLast(e, t) {
    return Ut(this, "findLast", e, t, Je, arguments);
  },
  findLastIndex(e, t) {
    return Ut(this, "findLastIndex", e, t, void 0, arguments);
  },
  // flat, flatMap could benefit from ARRAY_ITERATE but are not straight-forward to implement
  forEach(e, t) {
    return Ut(this, "forEach", e, t, void 0, arguments);
  },
  includes(...e) {
    return Vo(this, "includes", e);
  },
  indexOf(...e) {
    return Vo(this, "indexOf", e);
  },
  join(e) {
    return Nn(this).join(e);
  },
  // keys() iterator only reads `length`, no optimisation required
  lastIndexOf(...e) {
    return Vo(this, "lastIndexOf", e);
  },
  map(e, t) {
    return Ut(this, "map", e, t, void 0, arguments);
  },
  pop() {
    return Gn(this, "pop");
  },
  push(...e) {
    return Gn(this, "push", e);
  },
  reduce(e, ...t) {
    return hs(this, "reduce", e, t);
  },
  reduceRight(e, ...t) {
    return hs(this, "reduceRight", e, t);
  },
  shift() {
    return Gn(this, "shift");
  },
  // slice could use ARRAY_ITERATE but also seems to beg for range tracking
  some(e, t) {
    return Ut(this, "some", e, t, void 0, arguments);
  },
  splice(...e) {
    return Gn(this, "splice", e);
  },
  toReversed() {
    return Nn(this).toReversed();
  },
  toSorted(e) {
    return Nn(this).toSorted(e);
  },
  toSpliced(...e) {
    return Nn(this).toSpliced(...e);
  },
  unshift(...e) {
    return Gn(this, "unshift", e);
  },
  values() {
    return Uo(this, "values", Je);
  }
};
function Uo(e, t, n) {
  const r = Fi(e), o = r[t]();
  return r !== e && !Tt(e) && (o._next = o.next, o.next = () => {
    const i = o._next();
    return i.value && (i.value = n(i.value)), i;
  }), o;
}
const od = Array.prototype;
function Ut(e, t, n, r, o, i) {
  const s = Fi(e), a = s !== e && !Tt(e), l = s[t];
  if (l !== od[t]) {
    const d = l.apply(e, i);
    return a ? Je(d) : d;
  }
  let u = n;
  s !== e && (a ? u = function(d, f) {
    return n.call(this, Je(d), f, e);
  } : n.length > 2 && (u = function(d, f) {
    return n.call(this, d, f, e);
  }));
  const c = l.call(s, u, r);
  return a && o ? o(c) : c;
}
function hs(e, t, n, r) {
  const o = Fi(e);
  let i = n;
  return o !== e && (Tt(e) ? n.length > 3 && (i = function(s, a, l) {
    return n.call(this, s, a, l, e);
  }) : i = function(s, a, l) {
    return n.call(this, s, Je(a), l, e);
  }), o[t](i, ...r);
}
function Vo(e, t, n) {
  const r = le(e);
  Ue(r, "iterate", gr);
  const o = r[t](...n);
  return (o === -1 || o === !1) && Hi(n[0]) ? (n[0] = le(n[0]), r[t](...n)) : o;
}
function Gn(e, t, n = []) {
  Xt(), xi();
  const r = le(e)[t].apply(e, n);
  return Di(), Jt(), r;
}
const id = /* @__PURE__ */ ki("__proto__,__v_isRef,__isVue"), $l = new Set(
  /* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(qt)
);
function sd(e) {
  qt(e) || (e = String(e));
  const t = le(this);
  return Ue(t, "has", e), t.hasOwnProperty(e);
}
class Nl {
  constructor(t = !1, n = !1) {
    this._isReadonly = t, this._isShallow = n;
  }
  get(t, n, r) {
    if (n === "__v_skip") return t.__v_skip;
    const o = this._isReadonly, i = this._isShallow;
    if (n === "__v_isReactive")
      return !o;
    if (n === "__v_isReadonly")
      return o;
    if (n === "__v_isShallow")
      return i;
    if (n === "__v_raw")
      return r === (o ? i ? gd : Dl : i ? xl : Al).get(t) || // receiver is not the reactive proxy, but has the same prototype
      // this means the receiver is a user proxy of the reactive proxy
      Object.getPrototypeOf(t) === Object.getPrototypeOf(r) ? t : void 0;
    const s = X(t);
    if (!o) {
      let l;
      if (s && (l = rd[n]))
        return l;
      if (n === "hasOwnProperty")
        return sd;
    }
    const a = Reflect.get(
      t,
      n,
      // if this is a proxy wrapping a ref, return methods using the raw ref
      // as receiver so that we don't have to call `toRaw` on the ref in all
      // its class methods
      Me(t) ? t : r
    );
    return (qt(n) ? $l.has(n) : id(n)) || (o || Ue(t, "get", n), i) ? a : Me(a) ? s && Ii(n) ? a : a.value : Ee(a) ? o ? Ui(a) : To(a) : a;
  }
}
class Il extends Nl {
  constructor(t = !1) {
    super(!1, t);
  }
  set(t, n, r, o) {
    let i = t[n];
    if (!this._isShallow) {
      const l = Pn(i);
      if (!Tt(r) && !Pn(r) && (i = le(i), r = le(r)), !X(t) && Me(i) && !Me(r))
        return l ? !1 : (i.value = r, !0);
    }
    const s = X(t) && Ii(n) ? Number(n) < t.length : ce(t, n), a = Reflect.set(
      t,
      n,
      r,
      Me(t) ? t : o
    );
    return t === le(o) && (s ? ln(r, i) && Yt(t, "set", n, r) : Yt(t, "add", n, r)), a;
  }
  deleteProperty(t, n) {
    const r = ce(t, n);
    t[n];
    const o = Reflect.deleteProperty(t, n);
    return o && r && Yt(t, "delete", n, void 0), o;
  }
  has(t, n) {
    const r = Reflect.has(t, n);
    return (!qt(n) || !$l.has(n)) && Ue(t, "has", n), r;
  }
  ownKeys(t) {
    return Ue(
      t,
      "iterate",
      X(t) ? "length" : Cn
    ), Reflect.ownKeys(t);
  }
}
class ad extends Nl {
  constructor(t = !1) {
    super(!0, t);
  }
  set(t, n) {
    return !0;
  }
  deleteProperty(t, n) {
    return !0;
  }
}
const ld = /* @__PURE__ */ new Il(), ud = /* @__PURE__ */ new ad(), cd = /* @__PURE__ */ new Il(!0);
const ii = (e) => e, Br = (e) => Reflect.getPrototypeOf(e);
function dd(e, t, n) {
  return function(...r) {
    const o = this.__v_raw, i = le(o), s = jn(i), a = e === "entries" || e === Symbol.iterator && s, l = e === "keys" && s, u = o[e](...r), c = n ? ii : t ? si : Je;
    return !t && Ue(
      i,
      "iterate",
      l ? oi : Cn
    ), {
      // iterator protocol
      next() {
        const { value: d, done: f } = u.next();
        return f ? { value: d, done: f } : {
          value: a ? [c(d[0]), c(d[1])] : c(d),
          done: f
        };
      },
      // iterable protocol
      [Symbol.iterator]() {
        return this;
      }
    };
  };
}
function Kr(e) {
  return function(...t) {
    return e === "delete" ? !1 : e === "clear" ? void 0 : this;
  };
}
function fd(e, t) {
  const n = {
    get(o) {
      const i = this.__v_raw, s = le(i), a = le(o);
      e || (ln(o, a) && Ue(s, "get", o), Ue(s, "get", a));
      const { has: l } = Br(s), u = t ? ii : e ? si : Je;
      if (l.call(s, o))
        return u(i.get(o));
      if (l.call(s, a))
        return u(i.get(a));
      i !== s && i.get(o);
    },
    get size() {
      const o = this.__v_raw;
      return !e && Ue(le(o), "iterate", Cn), Reflect.get(o, "size", o);
    },
    has(o) {
      const i = this.__v_raw, s = le(i), a = le(o);
      return e || (ln(o, a) && Ue(s, "has", o), Ue(s, "has", a)), o === a ? i.has(o) : i.has(o) || i.has(a);
    },
    forEach(o, i) {
      const s = this, a = s.__v_raw, l = le(a), u = t ? ii : e ? si : Je;
      return !e && Ue(l, "iterate", Cn), a.forEach((c, d) => o.call(i, u(c), u(d), s));
    }
  };
  return Ie(
    n,
    e ? {
      add: Kr("add"),
      set: Kr("set"),
      delete: Kr("delete"),
      clear: Kr("clear")
    } : {
      add(o) {
        !t && !Tt(o) && !Pn(o) && (o = le(o));
        const i = le(this);
        return Br(i).has.call(i, o) || (i.add(o), Yt(i, "add", o, o)), this;
      },
      set(o, i) {
        !t && !Tt(i) && !Pn(i) && (i = le(i));
        const s = le(this), { has: a, get: l } = Br(s);
        let u = a.call(s, o);
        u || (o = le(o), u = a.call(s, o));
        const c = l.call(s, o);
        return s.set(o, i), u ? ln(i, c) && Yt(s, "set", o, i) : Yt(s, "add", o, i), this;
      },
      delete(o) {
        const i = le(this), { has: s, get: a } = Br(i);
        let l = s.call(i, o);
        l || (o = le(o), l = s.call(i, o)), a && a.call(i, o);
        const u = i.delete(o);
        return l && Yt(i, "delete", o, void 0), u;
      },
      clear() {
        const o = le(this), i = o.size !== 0, s = o.clear();
        return i && Yt(
          o,
          "clear",
          void 0,
          void 0
        ), s;
      }
    }
  ), [
    "keys",
    "values",
    "entries",
    Symbol.iterator
  ].forEach((o) => {
    n[o] = dd(o, e, t);
  }), n;
}
function ji(e, t) {
  const n = fd(e, t);
  return (r, o, i) => o === "__v_isReactive" ? !e : o === "__v_isReadonly" ? e : o === "__v_raw" ? r : Reflect.get(
    ce(n, o) && o in r ? n : r,
    o,
    i
  );
}
const pd = {
  get: /* @__PURE__ */ ji(!1, !1)
}, md = {
  get: /* @__PURE__ */ ji(!1, !0)
}, hd = {
  get: /* @__PURE__ */ ji(!0, !1)
};
const Al = /* @__PURE__ */ new WeakMap(), xl = /* @__PURE__ */ new WeakMap(), Dl = /* @__PURE__ */ new WeakMap(), gd = /* @__PURE__ */ new WeakMap();
function bd(e) {
  switch (e) {
    case "Object":
    case "Array":
      return 1;
    case "Map":
    case "Set":
    case "WeakMap":
    case "WeakSet":
      return 2;
    default:
      return 0;
  }
}
function vd(e) {
  return e.__v_skip || !Object.isExtensible(e) ? 0 : bd(Wc(e));
}
function To(e) {
  return Pn(e) ? e : Vi(
    e,
    !1,
    ld,
    pd,
    Al
  );
}
function yd(e) {
  return Vi(
    e,
    !1,
    cd,
    md,
    xl
  );
}
function Ui(e) {
  return Vi(
    e,
    !0,
    ud,
    hd,
    Dl
  );
}
function Vi(e, t, n, r, o) {
  if (!Ee(e) || e.__v_raw && !(t && e.__v_isReactive))
    return e;
  const i = vd(e);
  if (i === 0)
    return e;
  const s = o.get(e);
  if (s)
    return s;
  const a = new Proxy(
    e,
    i === 2 ? r : n
  );
  return o.set(e, a), a;
}
function sr(e) {
  return Pn(e) ? sr(e.__v_raw) : !!(e && e.__v_isReactive);
}
function Pn(e) {
  return !!(e && e.__v_isReadonly);
}
function Tt(e) {
  return !!(e && e.__v_isShallow);
}
function Hi(e) {
  return e ? !!e.__v_raw : !1;
}
function le(e) {
  const t = e && e.__v_raw;
  return t ? le(t) : e;
}
function Rl(e) {
  return !ce(e, "__v_skip") && Object.isExtensible(e) && ti(e, "__v_skip", !0), e;
}
const Je = (e) => Ee(e) ? To(e) : e, si = (e) => Ee(e) ? Ui(e) : e;
function Me(e) {
  return e ? e.__v_isRef === !0 : !1;
}
function De(e) {
  return Fl(e, !1);
}
function Ml(e) {
  return Fl(e, !0);
}
function Fl(e, t) {
  return Me(e) ? e : new _d(e, t);
}
class _d {
  constructor(t, n) {
    this.dep = new Mi(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = n ? t : le(t), this._value = n ? t : Je(t), this.__v_isShallow = n;
  }
  get value() {
    return this.dep.track(), this._value;
  }
  set value(t) {
    const n = this._rawValue, r = this.__v_isShallow || Tt(t) || Pn(t);
    t = r ? t : le(t), ln(t, n) && (this._rawValue = t, this._value = r ? t : Je(t), this.dep.trigger());
  }
}
function jl(e) {
  return Me(e) ? e.value : e;
}
const Sd = {
  get: (e, t, n) => t === "__v_raw" ? e : jl(Reflect.get(e, t, n)),
  set: (e, t, n, r) => {
    const o = e[t];
    return Me(o) && !Me(n) ? (o.value = n, !0) : Reflect.set(e, t, n, r);
  }
};
function Ul(e) {
  return sr(e) ? e : new Proxy(e, Sd);
}
class Ed {
  constructor(t, n, r) {
    this.fn = t, this.setter = n, this._value = void 0, this.dep = new Mi(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = hr - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !n, this.isSSR = r;
  }
  /**
   * @internal
   */
  notify() {
    if (this.flags |= 16, !(this.flags & 8) && // avoid infinite self recursion
    ge !== this)
      return Cl(this, !0), !0;
  }
  get value() {
    const t = this.dep.track();
    return Pl(this), t && (t.version = this.dep.version), this._value;
  }
  set value(t) {
    this.setter && this.setter(t);
  }
}
function Td(e, t, n = !1) {
  let r, o;
  return q(e) ? r = e : (r = e.get, o = e.set), new Ed(r, o, n);
}
const Yr = {}, so = /* @__PURE__ */ new WeakMap();
let Sn;
function Cd(e, t = !1, n = Sn) {
  if (n) {
    let r = so.get(n);
    r || so.set(n, r = []), r.push(e);
  }
}
function Od(e, t, n = me) {
  const { immediate: r, deep: o, once: i, scheduler: s, augmentJob: a, call: l } = n, u = (g) => o ? g : Tt(g) || o === !1 || o === 0 ? zt(g, 1) : zt(g);
  let c, d, f, h, _ = !1, E = !1;
  if (Me(e) ? (d = () => e.value, _ = Tt(e)) : sr(e) ? (d = () => u(e), _ = !0) : X(e) ? (E = !0, _ = e.some((g) => sr(g) || Tt(g)), d = () => e.map((g) => {
    if (Me(g))
      return g.value;
    if (sr(g))
      return u(g);
    if (q(g))
      return l ? l(g, 2) : g();
  })) : q(e) ? t ? d = l ? () => l(e, 2) : e : d = () => {
    if (f) {
      Xt();
      try {
        f();
      } finally {
        Jt();
      }
    }
    const g = Sn;
    Sn = c;
    try {
      return l ? l(e, 3, [h]) : e(h);
    } finally {
      Sn = g;
    }
  } : d = Dt, t && o) {
    const g = d, O = o === !0 ? 1 / 0 : o;
    d = () => zt(g(), O);
  }
  const w = ed(), P = () => {
    c.stop(), w && w.active && Ni(w.effects, c);
  };
  if (i && t) {
    const g = t;
    t = (...O) => {
      g(...O), P();
    };
  }
  let M = E ? new Array(e.length).fill(Yr) : Yr;
  const S = (g) => {
    if (!(!(c.flags & 1) || !c.dirty && !g))
      if (t) {
        const O = c.run();
        if (o || _ || (E ? O.some((L, A) => ln(L, M[A])) : ln(O, M))) {
          f && f();
          const L = Sn;
          Sn = c;
          try {
            const A = [
              O,
              // pass undefined as the old value when it's changed for the first time
              M === Yr ? void 0 : E && M[0] === Yr ? [] : M,
              h
            ];
            M = O, l ? l(t, 3, A) : (
              // @ts-expect-error
              t(...A)
            );
          } finally {
            Sn = L;
          }
        }
      } else
        c.run();
  };
  return a && a(S), c = new El(d), c.scheduler = s ? () => s(S, !1) : S, h = (g) => Cd(g, !1, c), f = c.onStop = () => {
    const g = so.get(c);
    if (g) {
      if (l)
        l(g, 4);
      else
        for (const O of g) O();
      so.delete(c);
    }
  }, t ? r ? S(!0) : M = c.run() : s ? s(S.bind(null, !0), !0) : c.run(), P.pause = c.pause.bind(c), P.resume = c.resume.bind(c), P.stop = P, P;
}
function zt(e, t = 1 / 0, n) {
  if (t <= 0 || !Ee(e) || e.__v_skip || (n = n || /* @__PURE__ */ new Set(), n.has(e)))
    return e;
  if (n.add(e), t--, Me(e))
    zt(e.value, t, n);
  else if (X(e))
    for (let r = 0; r < e.length; r++)
      zt(e[r], t, n);
  else if (pl(e) || jn(e))
    e.forEach((r) => {
      zt(r, t, n);
    });
  else if (gl(e)) {
    for (const r in e)
      zt(e[r], t, n);
    for (const r of Object.getOwnPropertySymbols(e))
      Object.prototype.propertyIsEnumerable.call(e, r) && zt(e[r], t, n);
  }
  return e;
}
/**
* @vue/runtime-core v3.5.18
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
function jr(e, t, n, r) {
  try {
    return r ? e(...r) : e();
  } catch (o) {
    Co(o, t, n);
  }
}
function Ot(e, t, n, r) {
  if (q(e)) {
    const o = jr(e, t, n, r);
    return o && ml(o) && o.catch((i) => {
      Co(i, t, n);
    }), o;
  }
  if (X(e)) {
    const o = [];
    for (let i = 0; i < e.length; i++)
      o.push(Ot(e[i], t, n, r));
    return o;
  }
}
function Co(e, t, n, r = !0) {
  const o = t ? t.vnode : null, { errorHandler: i, throwUnhandledErrorInProduction: s } = t && t.appContext.config || me;
  if (t) {
    let a = t.parent;
    const l = t.proxy, u = `https://vuejs.org/error-reference/#runtime-${n}`;
    for (; a; ) {
      const c = a.ec;
      if (c) {
        for (let d = 0; d < c.length; d++)
          if (c[d](e, l, u) === !1)
            return;
      }
      a = a.parent;
    }
    if (i) {
      Xt(), jr(i, null, 10, [
        e,
        l,
        u
      ]), Jt();
      return;
    }
  }
  Ld(e, n, o, r, s);
}
function Ld(e, t, n, r = !0, o = !1) {
  if (o)
    throw e;
  console.error(e);
}
const qe = [];
let Nt = -1;
const Un = [];
let nn = null, In = 0;
const Vl = /* @__PURE__ */ Promise.resolve();
let ao = null;
function Hl(e) {
  const t = ao || Vl;
  return e ? t.then(this ? e.bind(this) : e) : t;
}
function Pd(e) {
  let t = Nt + 1, n = qe.length;
  for (; t < n; ) {
    const r = t + n >>> 1, o = qe[r], i = br(o);
    i < e || i === e && o.flags & 2 ? t = r + 1 : n = r;
  }
  return t;
}
function Wi(e) {
  if (!(e.flags & 1)) {
    const t = br(e), n = qe[qe.length - 1];
    !n || // fast path when the job id is larger than the tail
    !(e.flags & 2) && t >= br(n) ? qe.push(e) : qe.splice(Pd(t), 0, e), e.flags |= 1, Wl();
  }
}
function Wl() {
  ao || (ao = Vl.then(Kl));
}
function wd(e) {
  X(e) ? Un.push(...e) : nn && e.id === -1 ? nn.splice(In + 1, 0, e) : e.flags & 1 || (Un.push(e), e.flags |= 1), Wl();
}
function gs(e, t, n = Nt + 1) {
  for (; n < qe.length; n++) {
    const r = qe[n];
    if (r && r.flags & 2) {
      if (e && r.id !== e.uid)
        continue;
      qe.splice(n, 1), n--, r.flags & 4 && (r.flags &= -2), r(), r.flags & 4 || (r.flags &= -2);
    }
  }
}
function Bl(e) {
  if (Un.length) {
    const t = [...new Set(Un)].sort(
      (n, r) => br(n) - br(r)
    );
    if (Un.length = 0, nn) {
      nn.push(...t);
      return;
    }
    for (nn = t, In = 0; In < nn.length; In++) {
      const n = nn[In];
      n.flags & 4 && (n.flags &= -2), n.flags & 8 || n(), n.flags &= -2;
    }
    nn = null, In = 0;
  }
}
const br = (e) => e.id == null ? e.flags & 2 ? -1 : 1 / 0 : e.id;
function Kl(e) {
  try {
    for (Nt = 0; Nt < qe.length; Nt++) {
      const t = qe[Nt];
      t && !(t.flags & 8) && (t.flags & 4 && (t.flags &= -2), jr(
        t,
        t.i,
        t.i ? 15 : 14
      ), t.flags & 4 || (t.flags &= -2));
    }
  } finally {
    for (; Nt < qe.length; Nt++) {
      const t = qe[Nt];
      t && (t.flags &= -2);
    }
    Nt = -1, qe.length = 0, Bl(), ao = null, (qe.length || Un.length) && Kl();
  }
}
let Re = null, Yl = null;
function lo(e) {
  const t = Re;
  return Re = e, Yl = e && e.type.__scopeId || null, t;
}
function Tn(e, t = Re, n) {
  if (!t || e._n)
    return e;
  const r = (...o) => {
    r._d && $s(-1);
    const i = lo(t);
    let s;
    try {
      s = e(...o);
    } finally {
      lo(i), r._d && $s(1);
    }
    return s;
  };
  return r._n = !0, r._c = !0, r._d = !0, r;
}
function zl(e, t) {
  if (Re === null)
    return e;
  const n = wo(Re), r = e.dirs || (e.dirs = []);
  for (let o = 0; o < t.length; o++) {
    let [i, s, a, l = me] = t[o];
    i && (q(i) && (i = {
      mounted: i,
      updated: i
    }), i.deep && zt(s), r.push({
      dir: i,
      instance: n,
      value: s,
      oldValue: void 0,
      arg: a,
      modifiers: l
    }));
  }
  return e;
}
function hn(e, t, n, r) {
  const o = e.dirs, i = t && t.dirs;
  for (let s = 0; s < o.length; s++) {
    const a = o[s];
    i && (a.oldValue = i[s].value);
    let l = a.dir[r];
    l && (Xt(), Ot(l, n, 8, [
      e.el,
      a,
      e,
      t
    ]), Jt());
  }
}
const Gl = Symbol("_vte"), Xl = (e) => e.__isTeleport, ar = (e) => e && (e.disabled || e.disabled === ""), bs = (e) => e && (e.defer || e.defer === ""), vs = (e) => typeof SVGElement < "u" && e instanceof SVGElement, ys = (e) => typeof MathMLElement == "function" && e instanceof MathMLElement, ai = (e, t) => {
  const n = e && e.to;
  return Oe(n) ? t ? t(n) : null : n;
}, Jl = {
  name: "Teleport",
  __isTeleport: !0,
  process(e, t, n, r, o, i, s, a, l, u) {
    const {
      mc: c,
      pc: d,
      pbc: f,
      o: { insert: h, querySelector: _, createText: E, createComment: w }
    } = u, P = ar(t.props);
    let { shapeFlag: M, children: S, dynamicChildren: g } = t;
    if (e == null) {
      const O = t.el = E(""), L = t.anchor = E("");
      h(O, n, r), h(L, n, r);
      const A = ($, B) => {
        M & 16 && (o && o.isCE && (o.ce._teleportTarget = $), c(
          S,
          $,
          B,
          o,
          i,
          s,
          a,
          l
        ));
      }, F = () => {
        const $ = t.target = ai(t.props, _), B = ql($, t, E, h);
        $ && (s !== "svg" && vs($) ? s = "svg" : s !== "mathml" && ys($) && (s = "mathml"), P || (A($, B), no(t, !1)));
      };
      P && (A(n, L), no(t, !0)), bs(t.props) ? (t.el.__isMounted = !1, Ge(() => {
        F(), delete t.el.__isMounted;
      }, i)) : F();
    } else {
      if (bs(t.props) && e.el.__isMounted === !1) {
        Ge(() => {
          Jl.process(
            e,
            t,
            n,
            r,
            o,
            i,
            s,
            a,
            l,
            u
          );
        }, i);
        return;
      }
      t.el = e.el, t.targetStart = e.targetStart;
      const O = t.anchor = e.anchor, L = t.target = e.target, A = t.targetAnchor = e.targetAnchor, F = ar(e.props), $ = F ? n : L, B = F ? O : A;
      if (s === "svg" || vs(L) ? s = "svg" : (s === "mathml" || ys(L)) && (s = "mathml"), g ? (f(
        e.dynamicChildren,
        g,
        $,
        o,
        i,
        s,
        a
      ), Ji(e, t, !0)) : l || d(
        e,
        t,
        $,
        B,
        o,
        i,
        s,
        a,
        !1
      ), P)
        F ? t.props && e.props && t.props.to !== e.props.to && (t.props.to = e.props.to) : zr(
          t,
          n,
          O,
          u,
          1
        );
      else if ((t.props && t.props.to) !== (e.props && e.props.to)) {
        const Y = t.target = ai(
          t.props,
          _
        );
        Y && zr(
          t,
          Y,
          null,
          u,
          0
        );
      } else F && zr(
        t,
        L,
        A,
        u,
        1
      );
      no(t, P);
    }
  },
  remove(e, t, n, { um: r, o: { remove: o } }, i) {
    const {
      shapeFlag: s,
      children: a,
      anchor: l,
      targetStart: u,
      targetAnchor: c,
      target: d,
      props: f
    } = e;
    if (d && (o(u), o(c)), i && o(l), s & 16) {
      const h = i || !ar(f);
      for (let _ = 0; _ < a.length; _++) {
        const E = a[_];
        r(
          E,
          t,
          n,
          h,
          !!E.dynamicChildren
        );
      }
    }
  },
  move: zr,
  hydrate: kd
};
function zr(e, t, n, { o: { insert: r }, m: o }, i = 2) {
  i === 0 && r(e.targetAnchor, t, n);
  const { el: s, anchor: a, shapeFlag: l, children: u, props: c } = e, d = i === 2;
  if (d && r(s, t, n), (!d || ar(c)) && l & 16)
    for (let f = 0; f < u.length; f++)
      o(
        u[f],
        t,
        n,
        2
      );
  d && r(a, t, n);
}
function kd(e, t, n, r, o, i, {
  o: { nextSibling: s, parentNode: a, querySelector: l, insert: u, createText: c }
}, d) {
  const f = t.target = ai(
    t.props,
    l
  );
  if (f) {
    const h = ar(t.props), _ = f._lpa || f.firstChild;
    if (t.shapeFlag & 16)
      if (h)
        t.anchor = d(
          s(e),
          t,
          a(e),
          n,
          r,
          o,
          i
        ), t.targetStart = _, t.targetAnchor = _ && s(_);
      else {
        t.anchor = s(e);
        let E = _;
        for (; E; ) {
          if (E && E.nodeType === 8) {
            if (E.data === "teleport start anchor")
              t.targetStart = E;
            else if (E.data === "teleport anchor") {
              t.targetAnchor = E, f._lpa = t.targetAnchor && s(t.targetAnchor);
              break;
            }
          }
          E = s(E);
        }
        t.targetAnchor || ql(f, t, c, u), d(
          _ && s(_),
          t,
          f,
          n,
          r,
          o,
          i
        );
      }
    no(t, h);
  }
  return t.anchor && s(t.anchor);
}
const $d = Jl;
function no(e, t) {
  const n = e.ctx;
  if (n && n.ut) {
    let r, o;
    for (t ? (r = e.el, o = e.anchor) : (r = e.targetStart, o = e.targetAnchor); r && r !== o; )
      r.nodeType === 1 && r.setAttribute("data-v-owner", n.uid), r = r.nextSibling;
    n.ut();
  }
}
function ql(e, t, n, r) {
  const o = t.targetStart = n(""), i = t.targetAnchor = n("");
  return o[Gl] = i, e && (r(o, e), r(i, e)), i;
}
const rn = Symbol("_leaveCb"), Gr = Symbol("_enterCb");
function Nd() {
  const e = {
    isMounted: !1,
    isLeaving: !1,
    isUnmounting: !1,
    leavingVNodes: /* @__PURE__ */ new Map()
  };
  return Vr(() => {
    e.isMounted = !0;
  }), Bi(() => {
    e.isUnmounting = !0;
  }), e;
}
const ut = [Function, Array], Zl = {
  mode: String,
  appear: Boolean,
  persisted: Boolean,
  // enter
  onBeforeEnter: ut,
  onEnter: ut,
  onAfterEnter: ut,
  onEnterCancelled: ut,
  // leave
  onBeforeLeave: ut,
  onLeave: ut,
  onAfterLeave: ut,
  onLeaveCancelled: ut,
  // appear
  onBeforeAppear: ut,
  onAppear: ut,
  onAfterAppear: ut,
  onAppearCancelled: ut
}, Ql = (e) => {
  const t = e.subTree;
  return t.component ? Ql(t.component) : t;
}, Id = {
  name: "BaseTransition",
  props: Zl,
  setup(e, { slots: t }) {
    const n = jt(), r = Nd();
    return () => {
      const o = t.default && nu(t.default(), !0);
      if (!o || !o.length)
        return;
      const i = eu(o), s = le(e), { mode: a } = s;
      if (r.isLeaving)
        return Ho(i);
      const l = _s(i);
      if (!l)
        return Ho(i);
      let u = li(
        l,
        s,
        r,
        n,
        // #11061, ensure enterHooks is fresh after clone
        (d) => u = d
      );
      l.type !== He && vr(l, u);
      let c = n.subTree && _s(n.subTree);
      if (c && c.type !== He && !En(l, c) && Ql(n).type !== He) {
        let d = li(
          c,
          s,
          r,
          n
        );
        if (vr(c, d), a === "out-in" && l.type !== He)
          return r.isLeaving = !0, d.afterLeave = () => {
            r.isLeaving = !1, n.job.flags & 8 || n.update(), delete d.afterLeave, c = void 0;
          }, Ho(i);
        a === "in-out" && l.type !== He ? d.delayLeave = (f, h, _) => {
          const E = tu(
            r,
            c
          );
          E[String(c.key)] = c, f[rn] = () => {
            h(), f[rn] = void 0, delete u.delayedLeave, c = void 0;
          }, u.delayedLeave = () => {
            _(), delete u.delayedLeave, c = void 0;
          };
        } : c = void 0;
      } else c && (c = void 0);
      return i;
    };
  }
};
function eu(e) {
  let t = e[0];
  if (e.length > 1) {
    for (const n of e)
      if (n.type !== He) {
        t = n;
        break;
      }
  }
  return t;
}
const Ad = Id;
function tu(e, t) {
  const { leavingVNodes: n } = e;
  let r = n.get(t.type);
  return r || (r = /* @__PURE__ */ Object.create(null), n.set(t.type, r)), r;
}
function li(e, t, n, r, o) {
  const {
    appear: i,
    mode: s,
    persisted: a = !1,
    onBeforeEnter: l,
    onEnter: u,
    onAfterEnter: c,
    onEnterCancelled: d,
    onBeforeLeave: f,
    onLeave: h,
    onAfterLeave: _,
    onLeaveCancelled: E,
    onBeforeAppear: w,
    onAppear: P,
    onAfterAppear: M,
    onAppearCancelled: S
  } = t, g = String(e.key), O = tu(n, e), L = ($, B) => {
    $ && Ot(
      $,
      r,
      9,
      B
    );
  }, A = ($, B) => {
    const Y = B[1];
    L($, B), X($) ? $.every((R) => R.length <= 1) && Y() : $.length <= 1 && Y();
  }, F = {
    mode: s,
    persisted: a,
    beforeEnter($) {
      let B = l;
      if (!n.isMounted)
        if (i)
          B = w || l;
        else
          return;
      $[rn] && $[rn](
        !0
        /* cancelled */
      );
      const Y = O[g];
      Y && En(e, Y) && Y.el[rn] && Y.el[rn](), L(B, [$]);
    },
    enter($) {
      let B = u, Y = c, R = d;
      if (!n.isMounted)
        if (i)
          B = P || u, Y = M || c, R = S || d;
        else
          return;
      let z = !1;
      const ae = $[Gr] = (Te) => {
        z || (z = !0, Te ? L(R, [$]) : L(Y, [$]), F.delayedLeave && F.delayedLeave(), $[Gr] = void 0);
      };
      B ? A(B, [$, ae]) : ae();
    },
    leave($, B) {
      const Y = String(e.key);
      if ($[Gr] && $[Gr](
        !0
        /* cancelled */
      ), n.isUnmounting)
        return B();
      L(f, [$]);
      let R = !1;
      const z = $[rn] = (ae) => {
        R || (R = !0, B(), ae ? L(E, [$]) : L(_, [$]), $[rn] = void 0, O[Y] === e && delete O[Y]);
      };
      O[Y] = e, h ? A(h, [$, z]) : z();
    },
    clone($) {
      const B = li(
        $,
        t,
        n,
        r,
        o
      );
      return o && o(B), B;
    }
  };
  return F;
}
function Ho(e) {
  if (Oo(e))
    return e = un(e), e.children = null, e;
}
function _s(e) {
  if (!Oo(e))
    return Xl(e.type) && e.children ? eu(e.children) : e;
  if (e.component)
    return e.component.subTree;
  const { shapeFlag: t, children: n } = e;
  if (n) {
    if (t & 16)
      return n[0];
    if (t & 32 && q(n.default))
      return n.default();
  }
}
function vr(e, t) {
  e.shapeFlag & 6 && e.component ? (e.transition = t, vr(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
function nu(e, t = !1, n) {
  let r = [], o = 0;
  for (let i = 0; i < e.length; i++) {
    let s = e[i];
    const a = n == null ? s.key : String(n) + String(s.key != null ? s.key : i);
    s.type === Ve ? (s.patchFlag & 128 && o++, r = r.concat(
      nu(s.children, t, a)
    )) : (t || s.type !== He) && r.push(a != null ? un(s, { key: a }) : s);
  }
  if (o > 1)
    for (let i = 0; i < r.length; i++)
      r[i].patchFlag = -2;
  return r;
}
/*! #__NO_SIDE_EFFECTS__ */
// @__NO_SIDE_EFFECTS__
function Ur(e, t) {
  return q(e) ? (
    // #8236: extend call and options.name access are considered side-effects
    // by Rollup, so we have to wrap it in a pure-annotated IIFE.
    Ie({ name: e.name }, t, { setup: e })
  ) : e;
}
function xd() {
  const e = jt();
  return e ? (e.appContext.config.idPrefix || "v") + "-" + e.ids[0] + e.ids[1]++ : "";
}
function ru(e) {
  e.ids = [e.ids[0] + e.ids[2]++ + "-", 0, 0];
}
function lr(e, t, n, r, o = !1) {
  if (X(e)) {
    e.forEach(
      (_, E) => lr(
        _,
        t && (X(t) ? t[E] : t),
        n,
        r,
        o
      )
    );
    return;
  }
  if (Vn(r) && !o) {
    r.shapeFlag & 512 && r.type.__asyncResolved && r.component.subTree.component && lr(e, t, n, r.component.subTree);
    return;
  }
  const i = r.shapeFlag & 4 ? wo(r.component) : r.el, s = o ? null : i, { i: a, r: l } = e, u = t && t.r, c = a.refs === me ? a.refs = {} : a.refs, d = a.setupState, f = le(d), h = d === me ? () => !1 : (_) => ce(f, _);
  if (u != null && u !== l && (Oe(u) ? (c[u] = null, h(u) && (d[u] = null)) : Me(u) && (u.value = null)), q(l))
    jr(l, a, 12, [s, c]);
  else {
    const _ = Oe(l), E = Me(l);
    if (_ || E) {
      const w = () => {
        if (e.f) {
          const P = _ ? h(l) ? d[l] : c[l] : l.value;
          o ? X(P) && Ni(P, i) : X(P) ? P.includes(i) || P.push(i) : _ ? (c[l] = [i], h(l) && (d[l] = c[l])) : (l.value = [i], e.k && (c[e.k] = l.value));
        } else _ ? (c[l] = s, h(l) && (d[l] = s)) : E && (l.value = s, e.k && (c[e.k] = s));
      };
      s ? (w.id = -1, Ge(w, n)) : w();
    }
  }
}
Eo().requestIdleCallback;
Eo().cancelIdleCallback;
const Vn = (e) => !!e.type.__asyncLoader, Oo = (e) => e.type.__isKeepAlive;
function Dd(e, t) {
  ou(e, "a", t);
}
function Rd(e, t) {
  ou(e, "da", t);
}
function ou(e, t, n = We) {
  const r = e.__wdc || (e.__wdc = () => {
    let o = n;
    for (; o; ) {
      if (o.isDeactivated)
        return;
      o = o.parent;
    }
    return e();
  });
  if (Lo(t, r, n), n) {
    let o = n.parent;
    for (; o && o.parent; )
      Oo(o.parent.vnode) && Md(r, t, n, o), o = o.parent;
  }
}
function Md(e, t, n, r) {
  const o = Lo(
    t,
    e,
    r,
    !0
    /* prepend */
  );
  Ki(() => {
    Ni(r[t], o);
  }, n);
}
function Lo(e, t, n = We, r = !1) {
  if (n) {
    const o = n[e] || (n[e] = []), i = t.__weh || (t.__weh = (...s) => {
      Xt();
      const a = Wr(n), l = Ot(t, n, e, s);
      return a(), Jt(), l;
    });
    return r ? o.unshift(i) : o.push(i), i;
  }
}
const Zt = (e) => (t, n = We) => {
  (!Sr || e === "sp") && Lo(e, (...r) => t(...r), n);
}, iu = Zt("bm"), Vr = Zt("m"), Fd = Zt(
  "bu"
), jd = Zt("u"), Bi = Zt(
  "bum"
), Ki = Zt("um"), Ud = Zt(
  "sp"
), Vd = Zt("rtg"), Hd = Zt("rtc");
function Wd(e, t = We) {
  Lo("ec", e, t);
}
const Yi = "components", Bd = "directives";
function uo(e, t) {
  return zi(Yi, e, !0, t) || e;
}
const su = Symbol.for("v-ndc");
function ui(e) {
  return Oe(e) ? zi(Yi, e, !1) || e : e || su;
}
function au(e) {
  return zi(Bd, e);
}
function zi(e, t, n = !0, r = !1) {
  const o = Re || We;
  if (o) {
    const i = o.type;
    if (e === Yi) {
      const a = Nf(
        i,
        !1
      );
      if (a && (a === t || a === pt(t) || a === So(pt(t))))
        return i;
    }
    const s = (
      // local registration
      // check instance[type] first which is resolved for options API
      Ss(o[e] || i[e], t) || // global registration
      Ss(o.appContext[e], t)
    );
    return !s && r ? i : s;
  }
}
function Ss(e, t) {
  return e && (e[t] || e[pt(t)] || e[So(pt(t))]);
}
function Xe(e, t, n = {}, r, o) {
  if (Re.ce || Re.parent && Vn(Re.parent) && Re.parent.ce)
    return t !== "default" && (n.name = t), be(), Ct(
      Ve,
      null,
      [ke("slot", n, r && r())],
      64
    );
  let i = e[t];
  i && i._c && (i._d = !1), be();
  const s = i && lu(i(n)), a = n.key || // slot content array of a dynamic conditional slot may have a branch
  // key attached in the `createSlots` helper, respect that
  s && s.key, l = Ct(
    Ve,
    {
      key: (a && !qt(a) ? a : `_${t}`) + // #7256 force differentiate fallback content from actual content
      (!s && r ? "_fb" : "")
    },
    s || (r ? r() : []),
    s && e._ === 1 ? 64 : -2
  );
  return l.scopeId && (l.slotScopeIds = [l.scopeId + "-s"]), i && i._c && (i._d = !0), l;
}
function lu(e) {
  return e.some((t) => _r(t) ? !(t.type === He || t.type === Ve && !lu(t.children)) : !0) ? e : null;
}
const ci = (e) => e ? Pu(e) ? wo(e) : ci(e.parent) : null, ur = (
  // Move PURE marker to new line to workaround compiler discarding it
  // due to type annotation
  /* @__PURE__ */ Ie(/* @__PURE__ */ Object.create(null), {
    $: (e) => e,
    $el: (e) => e.vnode.el,
    $data: (e) => e.data,
    $props: (e) => e.props,
    $attrs: (e) => e.attrs,
    $slots: (e) => e.slots,
    $refs: (e) => e.refs,
    $parent: (e) => ci(e.parent),
    $root: (e) => ci(e.root),
    $host: (e) => e.ce,
    $emit: (e) => e.emit,
    $options: (e) => cu(e),
    $forceUpdate: (e) => e.f || (e.f = () => {
      Wi(e.update);
    }),
    $nextTick: (e) => e.n || (e.n = Hl.bind(e.proxy)),
    $watch: (e) => pf.bind(e)
  })
), Wo = (e, t) => e !== me && !e.__isScriptSetup && ce(e, t), Kd = {
  get({ _: e }, t) {
    if (t === "__v_skip")
      return !0;
    const { ctx: n, setupState: r, data: o, props: i, accessCache: s, type: a, appContext: l } = e;
    let u;
    if (t[0] !== "$") {
      const h = s[t];
      if (h !== void 0)
        switch (h) {
          case 1:
            return r[t];
          case 2:
            return o[t];
          case 4:
            return n[t];
          case 3:
            return i[t];
        }
      else {
        if (Wo(r, t))
          return s[t] = 1, r[t];
        if (o !== me && ce(o, t))
          return s[t] = 2, o[t];
        if (
          // only cache other properties when instance has declared (thus stable)
          // props
          (u = e.propsOptions[0]) && ce(u, t)
        )
          return s[t] = 3, i[t];
        if (n !== me && ce(n, t))
          return s[t] = 4, n[t];
        di && (s[t] = 0);
      }
    }
    const c = ur[t];
    let d, f;
    if (c)
      return t === "$attrs" && Ue(e.attrs, "get", ""), c(e);
    if (
      // css module (injected by vue-loader)
      (d = a.__cssModules) && (d = d[t])
    )
      return d;
    if (n !== me && ce(n, t))
      return s[t] = 4, n[t];
    if (
      // global properties
      f = l.config.globalProperties, ce(f, t)
    )
      return f[t];
  },
  set({ _: e }, t, n) {
    const { data: r, setupState: o, ctx: i } = e;
    return Wo(o, t) ? (o[t] = n, !0) : r !== me && ce(r, t) ? (r[t] = n, !0) : ce(e.props, t) || t[0] === "$" && t.slice(1) in e ? !1 : (i[t] = n, !0);
  },
  has({
    _: { data: e, setupState: t, accessCache: n, ctx: r, appContext: o, propsOptions: i }
  }, s) {
    let a;
    return !!n[s] || e !== me && ce(e, s) || Wo(t, s) || (a = i[0]) && ce(a, s) || ce(r, s) || ce(ur, s) || ce(o.config.globalProperties, s);
  },
  defineProperty(e, t, n) {
    return n.get != null ? e._.accessCache[t] = 0 : ce(n, "value") && this.set(e, t, n.value, null), Reflect.defineProperty(e, t, n);
  }
};
function Es(e) {
  return X(e) ? e.reduce(
    (t, n) => (t[n] = null, t),
    {}
  ) : e;
}
let di = !0;
function Yd(e) {
  const t = cu(e), n = e.proxy, r = e.ctx;
  di = !1, t.beforeCreate && Ts(t.beforeCreate, e, "bc");
  const {
    // state
    data: o,
    computed: i,
    methods: s,
    watch: a,
    provide: l,
    inject: u,
    // lifecycle
    created: c,
    beforeMount: d,
    mounted: f,
    beforeUpdate: h,
    updated: _,
    activated: E,
    deactivated: w,
    beforeDestroy: P,
    beforeUnmount: M,
    destroyed: S,
    unmounted: g,
    render: O,
    renderTracked: L,
    renderTriggered: A,
    errorCaptured: F,
    serverPrefetch: $,
    // public API
    expose: B,
    inheritAttrs: Y,
    // assets
    components: R,
    directives: z,
    filters: ae
  } = t;
  if (u && zd(u, r, null), s)
    for (const te in s) {
      const Q = s[te];
      q(Q) && (r[te] = Q.bind(n));
    }
  if (o) {
    const te = o.call(n, n);
    Ee(te) && (e.data = To(te));
  }
  if (di = !0, i)
    for (const te in i) {
      const Q = i[te], Pe = q(Q) ? Q.bind(n, n) : q(Q.get) ? Q.get.bind(n, n) : Dt, we = !q(Q) && q(Q.set) ? Q.set.bind(n) : Dt, ue = dt({
        get: Pe,
        set: we
      });
      Object.defineProperty(r, te, {
        enumerable: !0,
        configurable: !0,
        get: () => ue.value,
        set: (he) => ue.value = he
      });
    }
  if (a)
    for (const te in a)
      uu(a[te], r, n, te);
  if (l) {
    const te = q(l) ? l.call(n) : l;
    Reflect.ownKeys(te).forEach((Q) => {
      Qd(Q, te[Q]);
    });
  }
  c && Ts(c, e, "c");
  function ne(te, Q) {
    X(Q) ? Q.forEach((Pe) => te(Pe.bind(n))) : Q && te(Q.bind(n));
  }
  if (ne(iu, d), ne(Vr, f), ne(Fd, h), ne(jd, _), ne(Dd, E), ne(Rd, w), ne(Wd, F), ne(Hd, L), ne(Vd, A), ne(Bi, M), ne(Ki, g), ne(Ud, $), X(B))
    if (B.length) {
      const te = e.exposed || (e.exposed = {});
      B.forEach((Q) => {
        Object.defineProperty(te, Q, {
          get: () => n[Q],
          set: (Pe) => n[Q] = Pe,
          enumerable: !0
        });
      });
    } else e.exposed || (e.exposed = {});
  O && e.render === Dt && (e.render = O), Y != null && (e.inheritAttrs = Y), R && (e.components = R), z && (e.directives = z), $ && ru(e);
}
function zd(e, t, n = Dt) {
  X(e) && (e = fi(e));
  for (const r in e) {
    const o = e[r];
    let i;
    Ee(o) ? "default" in o ? i = cr(
      o.from || r,
      o.default,
      !0
    ) : i = cr(o.from || r) : i = cr(o), Me(i) ? Object.defineProperty(t, r, {
      enumerable: !0,
      configurable: !0,
      get: () => i.value,
      set: (s) => i.value = s
    }) : t[r] = i;
  }
}
function Ts(e, t, n) {
  Ot(
    X(e) ? e.map((r) => r.bind(t.proxy)) : e.bind(t.proxy),
    t,
    n
  );
}
function uu(e, t, n, r) {
  let o = r.includes(".") ? Eu(n, r) : () => n[r];
  if (Oe(e)) {
    const i = t[e];
    q(i) && ft(o, i);
  } else if (q(e))
    ft(o, e.bind(n));
  else if (Ee(e))
    if (X(e))
      e.forEach((i) => uu(i, t, n, r));
    else {
      const i = q(e.handler) ? e.handler.bind(n) : t[e.handler];
      q(i) && ft(o, i, e);
    }
}
function cu(e) {
  const t = e.type, { mixins: n, extends: r } = t, {
    mixins: o,
    optionsCache: i,
    config: { optionMergeStrategies: s }
  } = e.appContext, a = i.get(t);
  let l;
  return a ? l = a : !o.length && !n && !r ? l = t : (l = {}, o.length && o.forEach(
    (u) => co(l, u, s, !0)
  ), co(l, t, s)), Ee(t) && i.set(t, l), l;
}
function co(e, t, n, r = !1) {
  const { mixins: o, extends: i } = t;
  i && co(e, i, n, !0), o && o.forEach(
    (s) => co(e, s, n, !0)
  );
  for (const s in t)
    if (!(r && s === "expose")) {
      const a = Gd[s] || n && n[s];
      e[s] = a ? a(e[s], t[s]) : t[s];
    }
  return e;
}
const Gd = {
  data: Cs,
  props: Os,
  emits: Os,
  // objects
  methods: tr,
  computed: tr,
  // lifecycle
  beforeCreate: Ye,
  created: Ye,
  beforeMount: Ye,
  mounted: Ye,
  beforeUpdate: Ye,
  updated: Ye,
  beforeDestroy: Ye,
  beforeUnmount: Ye,
  destroyed: Ye,
  unmounted: Ye,
  activated: Ye,
  deactivated: Ye,
  errorCaptured: Ye,
  serverPrefetch: Ye,
  // assets
  components: tr,
  directives: tr,
  // watch
  watch: Jd,
  // provide / inject
  provide: Cs,
  inject: Xd
};
function Cs(e, t) {
  return t ? e ? function() {
    return Ie(
      q(e) ? e.call(this, this) : e,
      q(t) ? t.call(this, this) : t
    );
  } : t : e;
}
function Xd(e, t) {
  return tr(fi(e), fi(t));
}
function fi(e) {
  if (X(e)) {
    const t = {};
    for (let n = 0; n < e.length; n++)
      t[e[n]] = e[n];
    return t;
  }
  return e;
}
function Ye(e, t) {
  return e ? [...new Set([].concat(e, t))] : t;
}
function tr(e, t) {
  return e ? Ie(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function Os(e, t) {
  return e ? X(e) && X(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : Ie(
    /* @__PURE__ */ Object.create(null),
    Es(e),
    Es(t ?? {})
  ) : t;
}
function Jd(e, t) {
  if (!e) return t;
  if (!t) return e;
  const n = Ie(/* @__PURE__ */ Object.create(null), e);
  for (const r in t)
    n[r] = Ye(e[r], t[r]);
  return n;
}
function du() {
  return {
    app: null,
    config: {
      isNativeTag: Vc,
      performance: !1,
      globalProperties: {},
      optionMergeStrategies: {},
      errorHandler: void 0,
      warnHandler: void 0,
      compilerOptions: {}
    },
    mixins: [],
    components: {},
    directives: {},
    provides: /* @__PURE__ */ Object.create(null),
    optionsCache: /* @__PURE__ */ new WeakMap(),
    propsCache: /* @__PURE__ */ new WeakMap(),
    emitsCache: /* @__PURE__ */ new WeakMap()
  };
}
let qd = 0;
function Zd(e, t) {
  return function(r, o = null) {
    q(r) || (r = Ie({}, r)), o != null && !Ee(o) && (o = null);
    const i = du(), s = /* @__PURE__ */ new WeakSet(), a = [];
    let l = !1;
    const u = i.app = {
      _uid: qd++,
      _component: r,
      _props: o,
      _container: null,
      _context: i,
      _instance: null,
      version: Af,
      get config() {
        return i.config;
      },
      set config(c) {
      },
      use(c, ...d) {
        return s.has(c) || (c && q(c.install) ? (s.add(c), c.install(u, ...d)) : q(c) && (s.add(c), c(u, ...d))), u;
      },
      mixin(c) {
        return i.mixins.includes(c) || i.mixins.push(c), u;
      },
      component(c, d) {
        return d ? (i.components[c] = d, u) : i.components[c];
      },
      directive(c, d) {
        return d ? (i.directives[c] = d, u) : i.directives[c];
      },
      mount(c, d, f) {
        if (!l) {
          const h = u._ceVNode || ke(r, o);
          return h.appContext = i, f === !0 ? f = "svg" : f === !1 && (f = void 0), e(h, c, f), l = !0, u._container = c, c.__vue_app__ = u, wo(h.component);
        }
      },
      onUnmount(c) {
        a.push(c);
      },
      unmount() {
        l && (Ot(
          a,
          u._instance,
          16
        ), e(null, u._container), delete u._container.__vue_app__);
      },
      provide(c, d) {
        return i.provides[c] = d, u;
      },
      runWithContext(c) {
        const d = Hn;
        Hn = u;
        try {
          return c();
        } finally {
          Hn = d;
        }
      }
    };
    return u;
  };
}
let Hn = null;
function Qd(e, t) {
  if (We) {
    let n = We.provides;
    const r = We.parent && We.parent.provides;
    r === n && (n = We.provides = Object.create(r)), n[e] = t;
  }
}
function cr(e, t, n = !1) {
  const r = jt();
  if (r || Hn) {
    let o = Hn ? Hn._context.provides : r ? r.parent == null || r.ce ? r.vnode.appContext && r.vnode.appContext.provides : r.parent.provides : void 0;
    if (o && e in o)
      return o[e];
    if (arguments.length > 1)
      return n && q(t) ? t.call(r && r.proxy) : t;
  }
}
const fu = {}, pu = () => Object.create(fu), mu = (e) => Object.getPrototypeOf(e) === fu;
function ef(e, t, n, r = !1) {
  const o = {}, i = pu();
  e.propsDefaults = /* @__PURE__ */ Object.create(null), hu(e, t, o, i);
  for (const s in e.propsOptions[0])
    s in o || (o[s] = void 0);
  n ? e.props = r ? o : yd(o) : e.type.props ? e.props = o : e.props = i, e.attrs = i;
}
function tf(e, t, n, r) {
  const {
    props: o,
    attrs: i,
    vnode: { patchFlag: s }
  } = e, a = le(o), [l] = e.propsOptions;
  let u = !1;
  if (
    // always force full diff in dev
    // - #1942 if hmr is enabled with sfc component
    // - vite#872 non-sfc component used by sfc component
    (r || s > 0) && !(s & 16)
  ) {
    if (s & 8) {
      const c = e.vnode.dynamicProps;
      for (let d = 0; d < c.length; d++) {
        let f = c[d];
        if (Po(e.emitsOptions, f))
          continue;
        const h = t[f];
        if (l)
          if (ce(i, f))
            h !== i[f] && (i[f] = h, u = !0);
          else {
            const _ = pt(f);
            o[_] = pi(
              l,
              a,
              _,
              h,
              e,
              !1
            );
          }
        else
          h !== i[f] && (i[f] = h, u = !0);
      }
    }
  } else {
    hu(e, t, o, i) && (u = !0);
    let c;
    for (const d in a)
      (!t || // for camelCase
      !ce(t, d) && // it's possible the original props was passed in as kebab-case
      // and converted to camelCase (#955)
      ((c = wn(d)) === d || !ce(t, c))) && (l ? n && // for camelCase
      (n[d] !== void 0 || // for kebab-case
      n[c] !== void 0) && (o[d] = pi(
        l,
        a,
        d,
        void 0,
        e,
        !0
      )) : delete o[d]);
    if (i !== a)
      for (const d in i)
        (!t || !ce(t, d)) && (delete i[d], u = !0);
  }
  u && Yt(e.attrs, "set", "");
}
function hu(e, t, n, r) {
  const [o, i] = e.propsOptions;
  let s = !1, a;
  if (t)
    for (let l in t) {
      if (rr(l))
        continue;
      const u = t[l];
      let c;
      o && ce(o, c = pt(l)) ? !i || !i.includes(c) ? n[c] = u : (a || (a = {}))[c] = u : Po(e.emitsOptions, l) || (!(l in r) || u !== r[l]) && (r[l] = u, s = !0);
    }
  if (i) {
    const l = le(n), u = a || me;
    for (let c = 0; c < i.length; c++) {
      const d = i[c];
      n[d] = pi(
        o,
        l,
        d,
        u[d],
        e,
        !ce(u, d)
      );
    }
  }
  return s;
}
function pi(e, t, n, r, o, i) {
  const s = e[n];
  if (s != null) {
    const a = ce(s, "default");
    if (a && r === void 0) {
      const l = s.default;
      if (s.type !== Function && !s.skipFactory && q(l)) {
        const { propsDefaults: u } = o;
        if (n in u)
          r = u[n];
        else {
          const c = Wr(o);
          r = u[n] = l.call(
            null,
            t
          ), c();
        }
      } else
        r = l;
      o.ce && o.ce._setProp(n, r);
    }
    s[
      0
      /* shouldCast */
    ] && (i && !a ? r = !1 : s[
      1
      /* shouldCastTrue */
    ] && (r === "" || r === wn(n)) && (r = !0));
  }
  return r;
}
const nf = /* @__PURE__ */ new WeakMap();
function gu(e, t, n = !1) {
  const r = n ? nf : t.propsCache, o = r.get(e);
  if (o)
    return o;
  const i = e.props, s = {}, a = [];
  let l = !1;
  if (!q(e)) {
    const c = (d) => {
      l = !0;
      const [f, h] = gu(d, t, !0);
      Ie(s, f), h && a.push(...h);
    };
    !n && t.mixins.length && t.mixins.forEach(c), e.extends && c(e.extends), e.mixins && e.mixins.forEach(c);
  }
  if (!i && !l)
    return Ee(e) && r.set(e, Fn), Fn;
  if (X(i))
    for (let c = 0; c < i.length; c++) {
      const d = pt(i[c]);
      Ls(d) && (s[d] = me);
    }
  else if (i)
    for (const c in i) {
      const d = pt(c);
      if (Ls(d)) {
        const f = i[c], h = s[d] = X(f) || q(f) ? { type: f } : Ie({}, f), _ = h.type;
        let E = !1, w = !0;
        if (X(_))
          for (let P = 0; P < _.length; ++P) {
            const M = _[P], S = q(M) && M.name;
            if (S === "Boolean") {
              E = !0;
              break;
            } else S === "String" && (w = !1);
          }
        else
          E = q(_) && _.name === "Boolean";
        h[
          0
          /* shouldCast */
        ] = E, h[
          1
          /* shouldCastTrue */
        ] = w, (E || ce(h, "default")) && a.push(d);
      }
    }
  const u = [s, a];
  return Ee(e) && r.set(e, u), u;
}
function Ls(e) {
  return e[0] !== "$" && !rr(e);
}
const Gi = (e) => e === "_" || e === "__" || e === "_ctx" || e === "$stable", Xi = (e) => X(e) ? e.map(It) : [It(e)], rf = (e, t, n) => {
  if (t._n)
    return t;
  const r = Tn((...o) => Xi(t(...o)), n);
  return r._c = !1, r;
}, bu = (e, t, n) => {
  const r = e._ctx;
  for (const o in e) {
    if (Gi(o)) continue;
    const i = e[o];
    if (q(i))
      t[o] = rf(o, i, r);
    else if (i != null) {
      const s = Xi(i);
      t[o] = () => s;
    }
  }
}, vu = (e, t) => {
  const n = Xi(t);
  e.slots.default = () => n;
}, yu = (e, t, n) => {
  for (const r in t)
    (n || !Gi(r)) && (e[r] = t[r]);
}, of = (e, t, n) => {
  const r = e.slots = pu();
  if (e.vnode.shapeFlag & 32) {
    const o = t.__;
    o && ti(r, "__", o, !0);
    const i = t._;
    i ? (yu(r, t, n), n && ti(r, "_", i, !0)) : bu(t, r);
  } else t && vu(e, t);
}, sf = (e, t, n) => {
  const { vnode: r, slots: o } = e;
  let i = !0, s = me;
  if (r.shapeFlag & 32) {
    const a = t._;
    a ? n && a === 1 ? i = !1 : yu(o, t, n) : (i = !t.$stable, bu(t, o)), s = t;
  } else t && (vu(e, t), s = { default: 1 });
  if (i)
    for (const a in o)
      !Gi(a) && s[a] == null && delete o[a];
}, Ge = _f;
function af(e) {
  return lf(e);
}
function lf(e, t) {
  const n = Eo();
  n.__VUE__ = !0;
  const {
    insert: r,
    remove: o,
    patchProp: i,
    createElement: s,
    createText: a,
    createComment: l,
    setText: u,
    setElementText: c,
    parentNode: d,
    nextSibling: f,
    setScopeId: h = Dt,
    insertStaticContent: _
  } = e, E = (b, y, v, N = null, x = null, D = null, V = void 0, U = null, p = !!y.dynamicChildren) => {
    if (b === y)
      return;
    b && !En(b, y) && (N = mt(b), he(b, x, D, !0), b = null), y.patchFlag === -2 && (p = !1, y.dynamicChildren = null);
    const { type: m, ref: T, shapeFlag: k } = y;
    switch (m) {
      case Hr:
        w(b, y, v, N);
        break;
      case He:
        P(b, y, v, N);
        break;
      case Ko:
        b == null && M(y, v, N, V);
        break;
      case Ve:
        R(
          b,
          y,
          v,
          N,
          x,
          D,
          V,
          U,
          p
        );
        break;
      default:
        k & 1 ? O(
          b,
          y,
          v,
          N,
          x,
          D,
          V,
          U,
          p
        ) : k & 6 ? z(
          b,
          y,
          v,
          N,
          x,
          D,
          V,
          U,
          p
        ) : (k & 64 || k & 128) && m.process(
          b,
          y,
          v,
          N,
          x,
          D,
          V,
          U,
          p,
          et
        );
    }
    T != null && x ? lr(T, b && b.ref, D, y || b, !y) : T == null && b && b.ref != null && lr(b.ref, null, D, b, !0);
  }, w = (b, y, v, N) => {
    if (b == null)
      r(
        y.el = a(y.children),
        v,
        N
      );
    else {
      const x = y.el = b.el;
      y.children !== b.children && u(x, y.children);
    }
  }, P = (b, y, v, N) => {
    b == null ? r(
      y.el = l(y.children || ""),
      v,
      N
    ) : y.el = b.el;
  }, M = (b, y, v, N) => {
    [b.el, b.anchor] = _(
      b.children,
      y,
      v,
      N,
      b.el,
      b.anchor
    );
  }, S = ({ el: b, anchor: y }, v, N) => {
    let x;
    for (; b && b !== y; )
      x = f(b), r(b, v, N), b = x;
    r(y, v, N);
  }, g = ({ el: b, anchor: y }) => {
    let v;
    for (; b && b !== y; )
      v = f(b), o(b), b = v;
    o(y);
  }, O = (b, y, v, N, x, D, V, U, p) => {
    y.type === "svg" ? V = "svg" : y.type === "math" && (V = "mathml"), b == null ? L(
      y,
      v,
      N,
      x,
      D,
      V,
      U,
      p
    ) : $(
      b,
      y,
      x,
      D,
      V,
      U,
      p
    );
  }, L = (b, y, v, N, x, D, V, U) => {
    let p, m;
    const { props: T, shapeFlag: k, transition: H, dirs: j } = b;
    if (p = b.el = s(
      b.type,
      D,
      T && T.is,
      T
    ), k & 8 ? c(p, b.children) : k & 16 && F(
      b.children,
      p,
      null,
      N,
      x,
      Bo(b, D),
      V,
      U
    ), j && hn(b, null, N, "created"), A(p, b, b.scopeId, V, N), T) {
      for (const I in T)
        I !== "value" && !rr(I) && i(p, I, null, T[I], D, N);
      "value" in T && i(p, "value", null, T.value, D), (m = T.onVnodeBeforeMount) && kt(m, N, b);
    }
    j && hn(b, null, N, "beforeMount");
    const C = uf(x, H);
    C && H.beforeEnter(p), r(p, y, v), ((m = T && T.onVnodeMounted) || C || j) && Ge(() => {
      m && kt(m, N, b), C && H.enter(p), j && hn(b, null, N, "mounted");
    }, x);
  }, A = (b, y, v, N, x) => {
    if (v && h(b, v), N)
      for (let D = 0; D < N.length; D++)
        h(b, N[D]);
    if (x) {
      let D = x.subTree;
      if (y === D || Cu(D.type) && (D.ssContent === y || D.ssFallback === y)) {
        const V = x.vnode;
        A(
          b,
          V,
          V.scopeId,
          V.slotScopeIds,
          x.parent
        );
      }
    }
  }, F = (b, y, v, N, x, D, V, U, p = 0) => {
    for (let m = p; m < b.length; m++) {
      const T = b[m] = U ? on(b[m]) : It(b[m]);
      E(
        null,
        T,
        y,
        v,
        N,
        x,
        D,
        V,
        U
      );
    }
  }, $ = (b, y, v, N, x, D, V) => {
    const U = y.el = b.el;
    let { patchFlag: p, dynamicChildren: m, dirs: T } = y;
    p |= b.patchFlag & 16;
    const k = b.props || me, H = y.props || me;
    let j;
    if (v && gn(v, !1), (j = H.onVnodeBeforeUpdate) && kt(j, v, y, b), T && hn(y, b, v, "beforeUpdate"), v && gn(v, !0), (k.innerHTML && H.innerHTML == null || k.textContent && H.textContent == null) && c(U, ""), m ? B(
      b.dynamicChildren,
      m,
      U,
      v,
      N,
      Bo(y, x),
      D
    ) : V || Q(
      b,
      y,
      U,
      null,
      v,
      N,
      Bo(y, x),
      D,
      !1
    ), p > 0) {
      if (p & 16)
        Y(U, k, H, v, x);
      else if (p & 2 && k.class !== H.class && i(U, "class", null, H.class, x), p & 4 && i(U, "style", k.style, H.style, x), p & 8) {
        const C = y.dynamicProps;
        for (let I = 0; I < C.length; I++) {
          const K = C[I], re = k[K], Ce = H[K];
          (Ce !== re || K === "value") && i(U, K, re, Ce, x, v);
        }
      }
      p & 1 && b.children !== y.children && c(U, y.children);
    } else !V && m == null && Y(U, k, H, v, x);
    ((j = H.onVnodeUpdated) || T) && Ge(() => {
      j && kt(j, v, y, b), T && hn(y, b, v, "updated");
    }, N);
  }, B = (b, y, v, N, x, D, V) => {
    for (let U = 0; U < y.length; U++) {
      const p = b[U], m = y[U], T = (
        // oldVNode may be an errored async setup() component inside Suspense
        // which will not have a mounted element
        p.el && // - In the case of a Fragment, we need to provide the actual parent
        // of the Fragment itself so it can move its children.
        (p.type === Ve || // - In the case of different nodes, there is going to be a replacement
        // which also requires the correct parent container
        !En(p, m) || // - In the case of a component, it could contain anything.
        p.shapeFlag & 198) ? d(p.el) : (
          // In other cases, the parent container is not actually used so we
          // just pass the block element here to avoid a DOM parentNode call.
          v
        )
      );
      E(
        p,
        m,
        T,
        null,
        N,
        x,
        D,
        V,
        !0
      );
    }
  }, Y = (b, y, v, N, x) => {
    if (y !== v) {
      if (y !== me)
        for (const D in y)
          !rr(D) && !(D in v) && i(
            b,
            D,
            y[D],
            null,
            x,
            N
          );
      for (const D in v) {
        if (rr(D)) continue;
        const V = v[D], U = y[D];
        V !== U && D !== "value" && i(b, D, U, V, x, N);
      }
      "value" in v && i(b, "value", y.value, v.value, x);
    }
  }, R = (b, y, v, N, x, D, V, U, p) => {
    const m = y.el = b ? b.el : a(""), T = y.anchor = b ? b.anchor : a("");
    let { patchFlag: k, dynamicChildren: H, slotScopeIds: j } = y;
    j && (U = U ? U.concat(j) : j), b == null ? (r(m, v, N), r(T, v, N), F(
      // #10007
      // such fragment like `<></>` will be compiled into
      // a fragment which doesn't have a children.
      // In this case fallback to an empty array
      y.children || [],
      v,
      T,
      x,
      D,
      V,
      U,
      p
    )) : k > 0 && k & 64 && H && // #2715 the previous fragment could've been a BAILed one as a result
    // of renderSlot() with no valid children
    b.dynamicChildren ? (B(
      b.dynamicChildren,
      H,
      v,
      x,
      D,
      V,
      U
    ), // #2080 if the stable fragment has a key, it's a <template v-for> that may
    //  get moved around. Make sure all root level vnodes inherit el.
    // #2134 or if it's a component root, it may also get moved around
    // as the component is being moved.
    (y.key != null || x && y === x.subTree) && Ji(
      b,
      y,
      !0
      /* shallow */
    )) : Q(
      b,
      y,
      v,
      T,
      x,
      D,
      V,
      U,
      p
    );
  }, z = (b, y, v, N, x, D, V, U, p) => {
    y.slotScopeIds = U, b == null ? y.shapeFlag & 512 ? x.ctx.activate(
      y,
      v,
      N,
      V,
      p
    ) : ae(
      y,
      v,
      N,
      x,
      D,
      V,
      p
    ) : Te(b, y, p);
  }, ae = (b, y, v, N, x, D, V) => {
    const U = b.component = Lf(
      b,
      N,
      x
    );
    if (Oo(b) && (U.ctx.renderer = et), Pf(U, !1, V), U.asyncDep) {
      if (x && x.registerDep(U, ne, V), !b.el) {
        const p = U.subTree = ke(He);
        P(null, p, y, v), b.placeholder = p.el;
      }
    } else
      ne(
        U,
        b,
        y,
        v,
        x,
        D,
        V
      );
  }, Te = (b, y, v) => {
    const N = y.component = b.component;
    if (vf(b, y, v))
      if (N.asyncDep && !N.asyncResolved) {
        te(N, y, v);
        return;
      } else
        N.next = y, N.update();
    else
      y.el = b.el, N.vnode = y;
  }, ne = (b, y, v, N, x, D, V) => {
    const U = () => {
      if (b.isMounted) {
        let { next: k, bu: H, u: j, parent: C, vnode: I } = b;
        {
          const Ae = _u(b);
          if (Ae) {
            k && (k.el = I.el, te(b, k, V)), Ae.asyncDep.then(() => {
              b.isUnmounted || U();
            });
            return;
          }
        }
        let K = k, re;
        gn(b, !1), k ? (k.el = I.el, te(b, k, V)) : k = I, H && Mo(H), (re = k.props && k.props.onVnodeBeforeUpdate) && kt(re, C, k, I), gn(b, !0);
        const Ce = ws(b), Ke = b.subTree;
        b.subTree = Ce, E(
          Ke,
          Ce,
          // parent may have changed if it's in a teleport
          d(Ke.el),
          // anchor may have changed if it's in a fragment
          mt(Ke),
          b,
          x,
          D
        ), k.el = Ce.el, K === null && yf(b, Ce.el), j && Ge(j, x), (re = k.props && k.props.onVnodeUpdated) && Ge(
          () => kt(re, C, k, I),
          x
        );
      } else {
        let k;
        const { el: H, props: j } = y, { bm: C, m: I, parent: K, root: re, type: Ce } = b, Ke = Vn(y);
        gn(b, !1), C && Mo(C), !Ke && (k = j && j.onVnodeBeforeMount) && kt(k, K, y), gn(b, !0);
        {
          re.ce && // @ts-expect-error _def is private
          re.ce._def.shadowRoot !== !1 && re.ce._injectChildStyle(Ce);
          const Ae = b.subTree = ws(b);
          E(
            null,
            Ae,
            v,
            N,
            b,
            x,
            D
          ), y.el = Ae.el;
        }
        if (I && Ge(I, x), !Ke && (k = j && j.onVnodeMounted)) {
          const Ae = y;
          Ge(
            () => kt(k, K, Ae),
            x
          );
        }
        (y.shapeFlag & 256 || K && Vn(K.vnode) && K.vnode.shapeFlag & 256) && b.a && Ge(b.a, x), b.isMounted = !0, y = v = N = null;
      }
    };
    b.scope.on();
    const p = b.effect = new El(U);
    b.scope.off();
    const m = b.update = p.run.bind(p), T = b.job = p.runIfDirty.bind(p);
    T.i = b, T.id = b.uid, p.scheduler = () => Wi(T), gn(b, !0), m();
  }, te = (b, y, v) => {
    y.component = b;
    const N = b.vnode.props;
    b.vnode = y, b.next = null, tf(b, y.props, N, v), sf(b, y.children, v), Xt(), gs(b), Jt();
  }, Q = (b, y, v, N, x, D, V, U, p = !1) => {
    const m = b && b.children, T = b ? b.shapeFlag : 0, k = y.children, { patchFlag: H, shapeFlag: j } = y;
    if (H > 0) {
      if (H & 128) {
        we(
          m,
          k,
          v,
          N,
          x,
          D,
          V,
          U,
          p
        );
        return;
      } else if (H & 256) {
        Pe(
          m,
          k,
          v,
          N,
          x,
          D,
          V,
          U,
          p
        );
        return;
      }
    }
    j & 8 ? (T & 16 && Qe(m, x, D), k !== m && c(v, k)) : T & 16 ? j & 16 ? we(
      m,
      k,
      v,
      N,
      x,
      D,
      V,
      U,
      p
    ) : Qe(m, x, D, !0) : (T & 8 && c(v, ""), j & 16 && F(
      k,
      v,
      N,
      x,
      D,
      V,
      U,
      p
    ));
  }, Pe = (b, y, v, N, x, D, V, U, p) => {
    b = b || Fn, y = y || Fn;
    const m = b.length, T = y.length, k = Math.min(m, T);
    let H;
    for (H = 0; H < k; H++) {
      const j = y[H] = p ? on(y[H]) : It(y[H]);
      E(
        b[H],
        j,
        v,
        null,
        x,
        D,
        V,
        U,
        p
      );
    }
    m > T ? Qe(
      b,
      x,
      D,
      !0,
      !1,
      k
    ) : F(
      y,
      v,
      N,
      x,
      D,
      V,
      U,
      p,
      k
    );
  }, we = (b, y, v, N, x, D, V, U, p) => {
    let m = 0;
    const T = y.length;
    let k = b.length - 1, H = T - 1;
    for (; m <= k && m <= H; ) {
      const j = b[m], C = y[m] = p ? on(y[m]) : It(y[m]);
      if (En(j, C))
        E(
          j,
          C,
          v,
          null,
          x,
          D,
          V,
          U,
          p
        );
      else
        break;
      m++;
    }
    for (; m <= k && m <= H; ) {
      const j = b[k], C = y[H] = p ? on(y[H]) : It(y[H]);
      if (En(j, C))
        E(
          j,
          C,
          v,
          null,
          x,
          D,
          V,
          U,
          p
        );
      else
        break;
      k--, H--;
    }
    if (m > k) {
      if (m <= H) {
        const j = H + 1, C = j < T ? y[j].el : N;
        for (; m <= H; )
          E(
            null,
            y[m] = p ? on(y[m]) : It(y[m]),
            v,
            C,
            x,
            D,
            V,
            U,
            p
          ), m++;
      }
    } else if (m > H)
      for (; m <= k; )
        he(b[m], x, D, !0), m++;
    else {
      const j = m, C = m, I = /* @__PURE__ */ new Map();
      for (m = C; m <= H; m++) {
        const st = y[m] = p ? on(y[m]) : It(y[m]);
        st.key != null && I.set(st.key, m);
      }
      let K, re = 0;
      const Ce = H - C + 1;
      let Ke = !1, Ae = 0;
      const mn = new Array(Ce);
      for (m = 0; m < Ce; m++) mn[m] = 0;
      for (m = j; m <= k; m++) {
        const st = b[m];
        if (re >= Ce) {
          he(st, x, D, !0);
          continue;
        }
        let wt;
        if (st.key != null)
          wt = I.get(st.key);
        else
          for (K = C; K <= H; K++)
            if (mn[K - C] === 0 && En(st, y[K])) {
              wt = K;
              break;
            }
        wt === void 0 ? he(st, x, D, !0) : (mn[wt - C] = m + 1, wt >= Ae ? Ae = wt : Ke = !0, E(
          st,
          y[wt],
          v,
          null,
          x,
          D,
          V,
          U,
          p
        ), re++);
      }
      const Do = Ke ? cf(mn) : Fn;
      for (K = Do.length - 1, m = Ce - 1; m >= 0; m--) {
        const st = C + m, wt = y[st], cs = y[st + 1], ds = st + 1 < T ? (
          // #13559, fallback to el placeholder for unresolved async component
          cs.el || cs.placeholder
        ) : N;
        mn[m] === 0 ? E(
          null,
          wt,
          v,
          ds,
          x,
          D,
          V,
          U,
          p
        ) : Ke && (K < 0 || m !== Do[K] ? ue(wt, v, ds, 2) : K--);
      }
    }
  }, ue = (b, y, v, N, x = null) => {
    const { el: D, type: V, transition: U, children: p, shapeFlag: m } = b;
    if (m & 6) {
      ue(b.component.subTree, y, v, N);
      return;
    }
    if (m & 128) {
      b.suspense.move(y, v, N);
      return;
    }
    if (m & 64) {
      V.move(b, y, v, et);
      return;
    }
    if (V === Ve) {
      r(D, y, v);
      for (let k = 0; k < p.length; k++)
        ue(p[k], y, v, N);
      r(b.anchor, y, v);
      return;
    }
    if (V === Ko) {
      S(b, y, v);
      return;
    }
    if (N !== 2 && m & 1 && U)
      if (N === 0)
        U.beforeEnter(D), r(D, y, v), Ge(() => U.enter(D), x);
      else {
        const { leave: k, delayLeave: H, afterLeave: j } = U, C = () => {
          b.ctx.isUnmounted ? o(D) : r(D, y, v);
        }, I = () => {
          k(D, () => {
            C(), j && j();
          });
        };
        H ? H(D, C, I) : I();
      }
    else
      r(D, y, v);
  }, he = (b, y, v, N = !1, x = !1) => {
    const {
      type: D,
      props: V,
      ref: U,
      children: p,
      dynamicChildren: m,
      shapeFlag: T,
      patchFlag: k,
      dirs: H,
      cacheIndex: j
    } = b;
    if (k === -2 && (x = !1), U != null && (Xt(), lr(U, null, v, b, !0), Jt()), j != null && (y.renderCache[j] = void 0), T & 256) {
      y.ctx.deactivate(b);
      return;
    }
    const C = T & 1 && H, I = !Vn(b);
    let K;
    if (I && (K = V && V.onVnodeBeforeUnmount) && kt(K, y, b), T & 6)
      Lt(b.component, v, N);
    else {
      if (T & 128) {
        b.suspense.unmount(v, N);
        return;
      }
      C && hn(b, null, y, "beforeUnmount"), T & 64 ? b.type.remove(
        b,
        y,
        v,
        et,
        N
      ) : m && // #5154
      // when v-once is used inside a block, setBlockTracking(-1) marks the
      // parent block with hasOnce: true
      // so that it doesn't take the fast path during unmount - otherwise
      // components nested in v-once are never unmounted.
      !m.hasOnce && // #1153: fast path should not be taken for non-stable (v-for) fragments
      (D !== Ve || k > 0 && k & 64) ? Qe(
        m,
        y,
        v,
        !1,
        !0
      ) : (D === Ve && k & 384 || !x && T & 16) && Qe(p, y, v), N && it(b);
    }
    (I && (K = V && V.onVnodeUnmounted) || C) && Ge(() => {
      K && kt(K, y, b), C && hn(b, null, y, "unmounted");
    }, v);
  }, it = (b) => {
    const { type: y, el: v, anchor: N, transition: x } = b;
    if (y === Ve) {
      Be(v, N);
      return;
    }
    if (y === Ko) {
      g(b);
      return;
    }
    const D = () => {
      o(v), x && !x.persisted && x.afterLeave && x.afterLeave();
    };
    if (b.shapeFlag & 1 && x && !x.persisted) {
      const { leave: V, delayLeave: U } = x, p = () => V(v, D);
      U ? U(b.el, D, p) : p();
    } else
      D();
  }, Be = (b, y) => {
    let v;
    for (; b !== y; )
      v = f(b), o(b), b = v;
    o(y);
  }, Lt = (b, y, v) => {
    const {
      bum: N,
      scope: x,
      job: D,
      subTree: V,
      um: U,
      m: p,
      a: m,
      parent: T,
      slots: { __: k }
    } = b;
    Ps(p), Ps(m), N && Mo(N), T && X(k) && k.forEach((H) => {
      T.renderCache[H] = void 0;
    }), x.stop(), D && (D.flags |= 8, he(V, b, y, v)), U && Ge(U, y), Ge(() => {
      b.isUnmounted = !0;
    }, y), y && y.pendingBranch && !y.isUnmounted && b.asyncDep && !b.asyncResolved && b.suspenseId === y.pendingId && (y.deps--, y.deps === 0 && y.resolve());
  }, Qe = (b, y, v, N = !1, x = !1, D = 0) => {
    for (let V = D; V < b.length; V++)
      he(b[V], y, v, N, x);
  }, mt = (b) => {
    if (b.shapeFlag & 6)
      return mt(b.component.subTree);
    if (b.shapeFlag & 128)
      return b.suspense.next();
    const y = f(b.anchor || b.el), v = y && y[Gl];
    return v ? f(v) : y;
  };
  let ht = !1;
  const Pt = (b, y, v) => {
    b == null ? y._vnode && he(y._vnode, null, null, !0) : E(
      y._vnode || null,
      b,
      y,
      null,
      null,
      null,
      v
    ), y._vnode = b, ht || (ht = !0, gs(), Bl(), ht = !1);
  }, et = {
    p: E,
    um: he,
    m: ue,
    r: it,
    mt: ae,
    mc: F,
    pc: Q,
    pbc: B,
    n: mt,
    o: e
  };
  return {
    render: Pt,
    hydrate: void 0,
    createApp: Zd(Pt)
  };
}
function Bo({ type: e, props: t }, n) {
  return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function gn({ effect: e, job: t }, n) {
  n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function uf(e, t) {
  return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function Ji(e, t, n = !1) {
  const r = e.children, o = t.children;
  if (X(r) && X(o))
    for (let i = 0; i < r.length; i++) {
      const s = r[i];
      let a = o[i];
      a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = o[i] = on(o[i]), a.el = s.el), !n && a.patchFlag !== -2 && Ji(s, a)), a.type === Hr && (a.el = s.el), a.type === He && !a.el && (a.el = s.el);
    }
}
function cf(e) {
  const t = e.slice(), n = [0];
  let r, o, i, s, a;
  const l = e.length;
  for (r = 0; r < l; r++) {
    const u = e[r];
    if (u !== 0) {
      if (o = n[n.length - 1], e[o] < u) {
        t[r] = o, n.push(r);
        continue;
      }
      for (i = 0, s = n.length - 1; i < s; )
        a = i + s >> 1, e[n[a]] < u ? i = a + 1 : s = a;
      u < e[n[i]] && (i > 0 && (t[r] = n[i - 1]), n[i] = r);
    }
  }
  for (i = n.length, s = n[i - 1]; i-- > 0; )
    n[i] = s, s = t[s];
  return n;
}
function _u(e) {
  const t = e.subTree.component;
  if (t)
    return t.asyncDep && !t.asyncResolved ? t : _u(t);
}
function Ps(e) {
  if (e)
    for (let t = 0; t < e.length; t++)
      e[t].flags |= 8;
}
const df = Symbol.for("v-scx"), ff = () => cr(df);
function ft(e, t, n) {
  return Su(e, t, n);
}
function Su(e, t, n = me) {
  const { immediate: r, deep: o, flush: i, once: s } = n, a = Ie({}, n), l = t && r || !t && i !== "post";
  let u;
  if (Sr) {
    if (i === "sync") {
      const h = ff();
      u = h.__watcherHandles || (h.__watcherHandles = []);
    } else if (!l) {
      const h = () => {
      };
      return h.stop = Dt, h.resume = Dt, h.pause = Dt, h;
    }
  }
  const c = We;
  a.call = (h, _, E) => Ot(h, c, _, E);
  let d = !1;
  i === "post" ? a.scheduler = (h) => {
    Ge(h, c && c.suspense);
  } : i !== "sync" && (d = !0, a.scheduler = (h, _) => {
    _ ? h() : Wi(h);
  }), a.augmentJob = (h) => {
    t && (h.flags |= 4), d && (h.flags |= 2, c && (h.id = c.uid, h.i = c));
  };
  const f = Od(e, t, a);
  return Sr && (u ? u.push(f) : l && f()), f;
}
function pf(e, t, n) {
  const r = this.proxy, o = Oe(e) ? e.includes(".") ? Eu(r, e) : () => r[e] : e.bind(r, r);
  let i;
  q(t) ? i = t : (i = t.handler, n = t);
  const s = Wr(this), a = Su(o, i.bind(r), n);
  return s(), a;
}
function Eu(e, t) {
  const n = t.split(".");
  return () => {
    let r = e;
    for (let o = 0; o < n.length && r; o++)
      r = r[n[o]];
    return r;
  };
}
const mf = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${pt(t)}Modifiers`] || e[`${wn(t)}Modifiers`];
function hf(e, t, ...n) {
  if (e.isUnmounted) return;
  const r = e.vnode.props || me;
  let o = n;
  const i = t.startsWith("update:"), s = i && mf(r, t.slice(7));
  s && (s.trim && (o = n.map((c) => Oe(c) ? c.trim() : c)), s.number && (o = n.map(Yc)));
  let a, l = r[a = Ro(t)] || // also try camelCase event handler (#2249)
  r[a = Ro(pt(t))];
  !l && i && (l = r[a = Ro(wn(t))]), l && Ot(
    l,
    e,
    6,
    o
  );
  const u = r[a + "Once"];
  if (u) {
    if (!e.emitted)
      e.emitted = {};
    else if (e.emitted[a])
      return;
    e.emitted[a] = !0, Ot(
      u,
      e,
      6,
      o
    );
  }
}
function Tu(e, t, n = !1) {
  const r = t.emitsCache, o = r.get(e);
  if (o !== void 0)
    return o;
  const i = e.emits;
  let s = {}, a = !1;
  if (!q(e)) {
    const l = (u) => {
      const c = Tu(u, t, !0);
      c && (a = !0, Ie(s, c));
    };
    !n && t.mixins.length && t.mixins.forEach(l), e.extends && l(e.extends), e.mixins && e.mixins.forEach(l);
  }
  return !i && !a ? (Ee(e) && r.set(e, null), null) : (X(i) ? i.forEach((l) => s[l] = null) : Ie(s, i), Ee(e) && r.set(e, s), s);
}
function Po(e, t) {
  return !e || !vo(t) ? !1 : (t = t.slice(2).replace(/Once$/, ""), ce(e, t[0].toLowerCase() + t.slice(1)) || ce(e, wn(t)) || ce(e, t));
}
function ws(e) {
  const {
    type: t,
    vnode: n,
    proxy: r,
    withProxy: o,
    propsOptions: [i],
    slots: s,
    attrs: a,
    emit: l,
    render: u,
    renderCache: c,
    props: d,
    data: f,
    setupState: h,
    ctx: _,
    inheritAttrs: E
  } = e, w = lo(e);
  let P, M;
  try {
    if (n.shapeFlag & 4) {
      const g = o || r, O = g;
      P = It(
        u.call(
          O,
          g,
          c,
          d,
          h,
          f,
          _
        )
      ), M = a;
    } else {
      const g = t;
      P = It(
        g.length > 1 ? g(
          d,
          { attrs: a, slots: s, emit: l }
        ) : g(
          d,
          null
        )
      ), M = t.props ? a : gf(a);
    }
  } catch (g) {
    dr.length = 0, Co(g, e, 1), P = ke(He);
  }
  let S = P;
  if (M && E !== !1) {
    const g = Object.keys(M), { shapeFlag: O } = S;
    g.length && O & 7 && (i && g.some($i) && (M = bf(
      M,
      i
    )), S = un(S, M, !1, !0));
  }
  return n.dirs && (S = un(S, null, !1, !0), S.dirs = S.dirs ? S.dirs.concat(n.dirs) : n.dirs), n.transition && vr(S, n.transition), P = S, lo(w), P;
}
const gf = (e) => {
  let t;
  for (const n in e)
    (n === "class" || n === "style" || vo(n)) && ((t || (t = {}))[n] = e[n]);
  return t;
}, bf = (e, t) => {
  const n = {};
  for (const r in e)
    (!$i(r) || !(r.slice(9) in t)) && (n[r] = e[r]);
  return n;
};
function vf(e, t, n) {
  const { props: r, children: o, component: i } = e, { props: s, children: a, patchFlag: l } = t, u = i.emitsOptions;
  if (t.dirs || t.transition)
    return !0;
  if (n && l >= 0) {
    if (l & 1024)
      return !0;
    if (l & 16)
      return r ? ks(r, s, u) : !!s;
    if (l & 8) {
      const c = t.dynamicProps;
      for (let d = 0; d < c.length; d++) {
        const f = c[d];
        if (s[f] !== r[f] && !Po(u, f))
          return !0;
      }
    }
  } else
    return (o || a) && (!a || !a.$stable) ? !0 : r === s ? !1 : r ? s ? ks(r, s, u) : !0 : !!s;
  return !1;
}
function ks(e, t, n) {
  const r = Object.keys(t);
  if (r.length !== Object.keys(e).length)
    return !0;
  for (let o = 0; o < r.length; o++) {
    const i = r[o];
    if (t[i] !== e[i] && !Po(n, i))
      return !0;
  }
  return !1;
}
function yf({ vnode: e, parent: t }, n) {
  for (; t; ) {
    const r = t.subTree;
    if (r.suspense && r.suspense.activeBranch === e && (r.el = e.el), r === e)
      (e = t.vnode).el = n, t = t.parent;
    else
      break;
  }
}
const Cu = (e) => e.__isSuspense;
function _f(e, t) {
  t && t.pendingBranch ? X(e) ? t.effects.push(...e) : t.effects.push(e) : wd(e);
}
const Ve = Symbol.for("v-fgt"), Hr = Symbol.for("v-txt"), He = Symbol.for("v-cmt"), Ko = Symbol.for("v-stc"), dr = [];
let at = null;
function be(e = !1) {
  dr.push(at = e ? null : []);
}
function Sf() {
  dr.pop(), at = dr[dr.length - 1] || null;
}
let yr = 1;
function $s(e, t = !1) {
  yr += e, e < 0 && at && t && (at.hasOnce = !0);
}
function Ou(e) {
  return e.dynamicChildren = yr > 0 ? at || Fn : null, Sf(), yr > 0 && at && at.push(e), e;
}
function Ze(e, t, n, r, o, i) {
  return Ou(
    Ft(
      e,
      t,
      n,
      r,
      o,
      i,
      !0
    )
  );
}
function Ct(e, t, n, r, o) {
  return Ou(
    ke(
      e,
      t,
      n,
      r,
      o,
      !0
    )
  );
}
function _r(e) {
  return e ? e.__v_isVNode === !0 : !1;
}
function En(e, t) {
  return e.type === t.type && e.key === t.key;
}
const Lu = ({ key: e }) => e ?? null, ro = ({
  ref: e,
  ref_key: t,
  ref_for: n
}) => (typeof e == "number" && (e = "" + e), e != null ? Oe(e) || Me(e) || q(e) ? { i: Re, r: e, k: t, f: !!n } : e : null);
function Ft(e, t = null, n = null, r = 0, o = null, i = e === Ve ? 0 : 1, s = !1, a = !1) {
  const l = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: e,
    props: t,
    key: t && Lu(t),
    ref: t && ro(t),
    scopeId: Yl,
    slotScopeIds: null,
    children: n,
    component: null,
    suspense: null,
    ssContent: null,
    ssFallback: null,
    dirs: null,
    transition: null,
    el: null,
    anchor: null,
    target: null,
    targetStart: null,
    targetAnchor: null,
    staticCount: 0,
    shapeFlag: i,
    patchFlag: r,
    dynamicProps: o,
    dynamicChildren: null,
    appContext: null,
    ctx: Re
  };
  return a ? (Zi(l, n), i & 128 && e.normalize(l)) : n && (l.shapeFlag |= Oe(n) ? 8 : 16), yr > 0 && // avoid a block node from tracking itself
  !s && // has current parent block
  at && // presence of a patch flag indicates this node needs patching on updates.
  // component nodes also should always be patched, because even if the
  // component doesn't need to update, it needs to persist the instance on to
  // the next vnode so that it can be properly unmounted later.
  (l.patchFlag > 0 || i & 6) && // the EVENTS flag is only for hydration and if it is the only flag, the
  // vnode should not be considered dynamic due to handler caching.
  l.patchFlag !== 32 && at.push(l), l;
}
const ke = Ef;
function Ef(e, t = null, n = null, r = 0, o = null, i = !1) {
  if ((!e || e === su) && (e = He), _r(e)) {
    const a = un(
      e,
      t,
      !0
      /* mergeRef: true */
    );
    return n && Zi(a, n), yr > 0 && !i && at && (a.shapeFlag & 6 ? at[at.indexOf(e)] = a : at.push(a)), a.patchFlag = -2, a;
  }
  if (If(e) && (e = e.__vccOpts), t) {
    t = Tf(t);
    let { class: a, style: l } = t;
    a && !Oe(a) && (t.class = Bn(a)), Ee(l) && (Hi(l) && !X(l) && (l = Ie({}, l)), t.style = Ai(l));
  }
  const s = Oe(e) ? 1 : Cu(e) ? 128 : Xl(e) ? 64 : Ee(e) ? 4 : q(e) ? 2 : 0;
  return Ft(
    e,
    t,
    n,
    r,
    o,
    s,
    i,
    !0
  );
}
function Tf(e) {
  return e ? Hi(e) || mu(e) ? Ie({}, e) : e : null;
}
function un(e, t, n = !1, r = !1) {
  const { props: o, ref: i, patchFlag: s, children: a, transition: l } = e, u = t ? de(o || {}, t) : o, c = {
    __v_isVNode: !0,
    __v_skip: !0,
    type: e.type,
    props: u,
    key: u && Lu(u),
    ref: t && t.ref ? (
      // #2078 in the case of <component :is="vnode" ref="extra"/>
      // if the vnode itself already has a ref, cloneVNode will need to merge
      // the refs so the single vnode can be set on multiple refs
      n && i ? X(i) ? i.concat(ro(t)) : [i, ro(t)] : ro(t)
    ) : i,
    scopeId: e.scopeId,
    slotScopeIds: e.slotScopeIds,
    children: a,
    target: e.target,
    targetStart: e.targetStart,
    targetAnchor: e.targetAnchor,
    staticCount: e.staticCount,
    shapeFlag: e.shapeFlag,
    // if the vnode is cloned with extra props, we can no longer assume its
    // existing patch flag to be reliable and need to add the FULL_PROPS flag.
    // note: preserve flag for fragments since they use the flag for children
    // fast paths only.
    patchFlag: t && e.type !== Ve ? s === -1 ? 16 : s | 16 : s,
    dynamicProps: e.dynamicProps,
    dynamicChildren: e.dynamicChildren,
    appContext: e.appContext,
    dirs: e.dirs,
    transition: l,
    // These should technically only be non-null on mounted VNodes. However,
    // they *should* be copied for kept-alive vnodes. So we just always copy
    // them since them being non-null during a mount doesn't affect the logic as
    // they will simply be overwritten.
    component: e.component,
    suspense: e.suspense,
    ssContent: e.ssContent && un(e.ssContent),
    ssFallback: e.ssFallback && un(e.ssFallback),
    placeholder: e.placeholder,
    el: e.el,
    anchor: e.anchor,
    ctx: e.ctx,
    ce: e.ce
  };
  return l && r && vr(
    c,
    l.clone(c)
  ), c;
}
function qi(e = " ", t = 0) {
  return ke(Hr, null, e, t);
}
function vt(e = "", t = !1) {
  return t ? (be(), Ct(He, null, e)) : ke(He, null, e);
}
function It(e) {
  return e == null || typeof e == "boolean" ? ke(He) : X(e) ? ke(
    Ve,
    null,
    // #3666, avoid reference pollution when reusing vnode
    e.slice()
  ) : _r(e) ? on(e) : ke(Hr, null, String(e));
}
function on(e) {
  return e.el === null && e.patchFlag !== -1 || e.memo ? e : un(e);
}
function Zi(e, t) {
  let n = 0;
  const { shapeFlag: r } = e;
  if (t == null)
    t = null;
  else if (X(t))
    n = 16;
  else if (typeof t == "object")
    if (r & 65) {
      const o = t.default;
      o && (o._c && (o._d = !1), Zi(e, o()), o._c && (o._d = !0));
      return;
    } else {
      n = 32;
      const o = t._;
      !o && !mu(t) ? t._ctx = Re : o === 3 && Re && (Re.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
    }
  else q(t) ? (t = { default: t, _ctx: Re }, n = 32) : (t = String(t), r & 64 ? (n = 16, t = [qi(t)]) : n = 8);
  e.children = t, e.shapeFlag |= n;
}
function de(...e) {
  const t = {};
  for (let n = 0; n < e.length; n++) {
    const r = e[n];
    for (const o in r)
      if (o === "class")
        t.class !== r.class && (t.class = Bn([t.class, r.class]));
      else if (o === "style")
        t.style = Ai([t.style, r.style]);
      else if (vo(o)) {
        const i = t[o], s = r[o];
        s && i !== s && !(X(i) && i.includes(s)) && (t[o] = i ? [].concat(i, s) : s);
      } else o !== "" && (t[o] = r[o]);
  }
  return t;
}
function kt(e, t, n, r = null) {
  Ot(e, t, 7, [
    n,
    r
  ]);
}
const Cf = du();
let Of = 0;
function Lf(e, t, n) {
  const r = e.type, o = (t ? t.appContext : e.appContext) || Cf, i = {
    uid: Of++,
    vnode: e,
    type: r,
    parent: t,
    appContext: o,
    root: null,
    // to be immediately set
    next: null,
    subTree: null,
    // will be set synchronously right after creation
    effect: null,
    update: null,
    // will be set synchronously right after creation
    job: null,
    scope: new _l(
      !0
      /* detached */
    ),
    render: null,
    proxy: null,
    exposed: null,
    exposeProxy: null,
    withProxy: null,
    provides: t ? t.provides : Object.create(o.provides),
    ids: t ? t.ids : ["", 0, 0],
    accessCache: null,
    renderCache: [],
    // local resolved assets
    components: null,
    directives: null,
    // resolved props and emits options
    propsOptions: gu(r, o),
    emitsOptions: Tu(r, o),
    // emit
    emit: null,
    // to be set immediately
    emitted: null,
    // props default value
    propsDefaults: me,
    // inheritAttrs
    inheritAttrs: r.inheritAttrs,
    // state
    ctx: me,
    data: me,
    props: me,
    attrs: me,
    slots: me,
    refs: me,
    setupState: me,
    setupContext: null,
    // suspense related
    suspense: n,
    suspenseId: n ? n.pendingId : 0,
    asyncDep: null,
    asyncResolved: !1,
    // lifecycle hooks
    // not using enums here because it results in computed properties
    isMounted: !1,
    isUnmounted: !1,
    isDeactivated: !1,
    bc: null,
    c: null,
    bm: null,
    m: null,
    bu: null,
    u: null,
    um: null,
    bum: null,
    da: null,
    a: null,
    rtg: null,
    rtc: null,
    ec: null,
    sp: null
  };
  return i.ctx = { _: i }, i.root = t ? t.root : i, i.emit = hf.bind(null, i), e.ce && e.ce(i), i;
}
let We = null;
const jt = () => We || Re;
let fo, mi;
{
  const e = Eo(), t = (n, r) => {
    let o;
    return (o = e[n]) || (o = e[n] = []), o.push(r), (i) => {
      o.length > 1 ? o.forEach((s) => s(i)) : o[0](i);
    };
  };
  fo = t(
    "__VUE_INSTANCE_SETTERS__",
    (n) => We = n
  ), mi = t(
    "__VUE_SSR_SETTERS__",
    (n) => Sr = n
  );
}
const Wr = (e) => {
  const t = We;
  return fo(e), e.scope.on(), () => {
    e.scope.off(), fo(t);
  };
}, Ns = () => {
  We && We.scope.off(), fo(null);
};
function Pu(e) {
  return e.vnode.shapeFlag & 4;
}
let Sr = !1;
function Pf(e, t = !1, n = !1) {
  t && mi(t);
  const { props: r, children: o } = e.vnode, i = Pu(e);
  ef(e, r, i, t), of(e, o, n || t);
  const s = i ? wf(e, t) : void 0;
  return t && mi(!1), s;
}
function wf(e, t) {
  const n = e.type;
  e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Kd);
  const { setup: r } = n;
  if (r) {
    Xt();
    const o = e.setupContext = r.length > 1 ? $f(e) : null, i = Wr(e), s = jr(
      r,
      e,
      0,
      [
        e.props,
        o
      ]
    ), a = ml(s);
    if (Jt(), i(), (a || e.sp) && !Vn(e) && ru(e), a) {
      if (s.then(Ns, Ns), t)
        return s.then((l) => {
          Is(e, l);
        }).catch((l) => {
          Co(l, e, 0);
        });
      e.asyncDep = s;
    } else
      Is(e, s);
  } else
    wu(e);
}
function Is(e, t, n) {
  q(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : Ee(t) && (e.setupState = Ul(t)), wu(e);
}
function wu(e, t, n) {
  const r = e.type;
  e.render || (e.render = r.render || Dt);
  {
    const o = Wr(e);
    Xt();
    try {
      Yd(e);
    } finally {
      Jt(), o();
    }
  }
}
const kf = {
  get(e, t) {
    return Ue(e, "get", ""), e[t];
  }
};
function $f(e) {
  const t = (n) => {
    e.exposed = n || {};
  };
  return {
    attrs: new Proxy(e.attrs, kf),
    slots: e.slots,
    emit: e.emit,
    expose: t
  };
}
function wo(e) {
  return e.exposed ? e.exposeProxy || (e.exposeProxy = new Proxy(Ul(Rl(e.exposed)), {
    get(t, n) {
      if (n in t)
        return t[n];
      if (n in ur)
        return ur[n](e);
    },
    has(t, n) {
      return n in t || n in ur;
    }
  })) : e.proxy;
}
function Nf(e, t = !0) {
  return q(e) ? e.displayName || e.name : e.name || t && e.__name;
}
function If(e) {
  return q(e) && "__vccOpts" in e;
}
const dt = (e, t) => Td(e, t, Sr);
function Qi(e, t, n) {
  const r = arguments.length;
  return r === 2 ? Ee(t) && !X(t) ? _r(t) ? ke(e, null, [t]) : ke(e, t) : ke(e, null, t) : (r > 3 ? n = Array.prototype.slice.call(arguments, 2) : r === 3 && _r(n) && (n = [n]), ke(e, t, n));
}
const Af = "3.5.18";
/**
* @vue/runtime-dom v3.5.18
* (c) 2018-present Yuxi (Evan) You and Vue contributors
* @license MIT
**/
let hi;
const As = typeof window < "u" && window.trustedTypes;
if (As)
  try {
    hi = /* @__PURE__ */ As.createPolicy("vue", {
      createHTML: (e) => e
    });
  } catch {
  }
const ku = hi ? (e) => hi.createHTML(e) : (e) => e, xf = "http://www.w3.org/2000/svg", Df = "http://www.w3.org/1998/Math/MathML", Kt = typeof document < "u" ? document : null, xs = Kt && /* @__PURE__ */ Kt.createElement("template"), Rf = {
  insert: (e, t, n) => {
    t.insertBefore(e, n || null);
  },
  remove: (e) => {
    const t = e.parentNode;
    t && t.removeChild(e);
  },
  createElement: (e, t, n, r) => {
    const o = t === "svg" ? Kt.createElementNS(xf, e) : t === "mathml" ? Kt.createElementNS(Df, e) : n ? Kt.createElement(e, { is: n }) : Kt.createElement(e);
    return e === "select" && r && r.multiple != null && o.setAttribute("multiple", r.multiple), o;
  },
  createText: (e) => Kt.createTextNode(e),
  createComment: (e) => Kt.createComment(e),
  setText: (e, t) => {
    e.nodeValue = t;
  },
  setElementText: (e, t) => {
    e.textContent = t;
  },
  parentNode: (e) => e.parentNode,
  nextSibling: (e) => e.nextSibling,
  querySelector: (e) => Kt.querySelector(e),
  setScopeId(e, t) {
    e.setAttribute(t, "");
  },
  // __UNSAFE__
  // Reason: innerHTML.
  // Static content here can only come from compiled templates.
  // As long as the user only uses trusted templates, this is safe.
  insertStaticContent(e, t, n, r, o, i) {
    const s = n ? n.previousSibling : t.lastChild;
    if (o && (o === i || o.nextSibling))
      for (; t.insertBefore(o.cloneNode(!0), n), !(o === i || !(o = o.nextSibling)); )
        ;
    else {
      xs.innerHTML = ku(
        r === "svg" ? `<svg>${e}</svg>` : r === "mathml" ? `<math>${e}</math>` : e
      );
      const a = xs.content;
      if (r === "svg" || r === "mathml") {
        const l = a.firstChild;
        for (; l.firstChild; )
          a.appendChild(l.firstChild);
        a.removeChild(l);
      }
      t.insertBefore(a, n);
    }
    return [
      // first
      s ? s.nextSibling : t.firstChild,
      // last
      n ? n.previousSibling : t.lastChild
    ];
  }
}, en = "transition", Xn = "animation", Er = Symbol("_vtc"), $u = {
  name: String,
  type: String,
  css: {
    type: Boolean,
    default: !0
  },
  duration: [String, Number, Object],
  enterFromClass: String,
  enterActiveClass: String,
  enterToClass: String,
  appearFromClass: String,
  appearActiveClass: String,
  appearToClass: String,
  leaveFromClass: String,
  leaveActiveClass: String,
  leaveToClass: String
}, Mf = /* @__PURE__ */ Ie(
  {},
  Zl,
  $u
), Ff = (e) => (e.displayName = "Transition", e.props = Mf, e), jf = /* @__PURE__ */ Ff(
  (e, { slots: t }) => Qi(Ad, Uf(e), t)
), bn = (e, t = []) => {
  X(e) ? e.forEach((n) => n(...t)) : e && e(...t);
}, Ds = (e) => e ? X(e) ? e.some((t) => t.length > 1) : e.length > 1 : !1;
function Uf(e) {
  const t = {};
  for (const R in e)
    R in $u || (t[R] = e[R]);
  if (e.css === !1)
    return t;
  const {
    name: n = "v",
    type: r,
    duration: o,
    enterFromClass: i = `${n}-enter-from`,
    enterActiveClass: s = `${n}-enter-active`,
    enterToClass: a = `${n}-enter-to`,
    appearFromClass: l = i,
    appearActiveClass: u = s,
    appearToClass: c = a,
    leaveFromClass: d = `${n}-leave-from`,
    leaveActiveClass: f = `${n}-leave-active`,
    leaveToClass: h = `${n}-leave-to`
  } = e, _ = Vf(o), E = _ && _[0], w = _ && _[1], {
    onBeforeEnter: P,
    onEnter: M,
    onEnterCancelled: S,
    onLeave: g,
    onLeaveCancelled: O,
    onBeforeAppear: L = P,
    onAppear: A = M,
    onAppearCancelled: F = S
  } = t, $ = (R, z, ae, Te) => {
    R._enterCancelled = Te, vn(R, z ? c : a), vn(R, z ? u : s), ae && ae();
  }, B = (R, z) => {
    R._isLeaving = !1, vn(R, d), vn(R, h), vn(R, f), z && z();
  }, Y = (R) => (z, ae) => {
    const Te = R ? A : M, ne = () => $(z, R, ae);
    bn(Te, [z, ne]), Rs(() => {
      vn(z, R ? l : i), Vt(z, R ? c : a), Ds(Te) || Ms(z, r, E, ne);
    });
  };
  return Ie(t, {
    onBeforeEnter(R) {
      bn(P, [R]), Vt(R, i), Vt(R, s);
    },
    onBeforeAppear(R) {
      bn(L, [R]), Vt(R, l), Vt(R, u);
    },
    onEnter: Y(!1),
    onAppear: Y(!0),
    onLeave(R, z) {
      R._isLeaving = !0;
      const ae = () => B(R, z);
      Vt(R, d), R._enterCancelled ? (Vt(R, f), Us()) : (Us(), Vt(R, f)), Rs(() => {
        R._isLeaving && (vn(R, d), Vt(R, h), Ds(g) || Ms(R, r, w, ae));
      }), bn(g, [R, ae]);
    },
    onEnterCancelled(R) {
      $(R, !1, void 0, !0), bn(S, [R]);
    },
    onAppearCancelled(R) {
      $(R, !0, void 0, !0), bn(F, [R]);
    },
    onLeaveCancelled(R) {
      B(R), bn(O, [R]);
    }
  });
}
function Vf(e) {
  if (e == null)
    return null;
  if (Ee(e))
    return [Yo(e.enter), Yo(e.leave)];
  {
    const t = Yo(e);
    return [t, t];
  }
}
function Yo(e) {
  return zc(e);
}
function Vt(e, t) {
  t.split(/\s+/).forEach((n) => n && e.classList.add(n)), (e[Er] || (e[Er] = /* @__PURE__ */ new Set())).add(t);
}
function vn(e, t) {
  t.split(/\s+/).forEach((r) => r && e.classList.remove(r));
  const n = e[Er];
  n && (n.delete(t), n.size || (e[Er] = void 0));
}
function Rs(e) {
  requestAnimationFrame(() => {
    requestAnimationFrame(e);
  });
}
let Hf = 0;
function Ms(e, t, n, r) {
  const o = e._endId = ++Hf, i = () => {
    o === e._endId && r();
  };
  if (n != null)
    return setTimeout(i, n);
  const { type: s, timeout: a, propCount: l } = Wf(e, t);
  if (!s)
    return r();
  const u = s + "end";
  let c = 0;
  const d = () => {
    e.removeEventListener(u, f), i();
  }, f = (h) => {
    h.target === e && ++c >= l && d();
  };
  setTimeout(() => {
    c < l && d();
  }, a + 1), e.addEventListener(u, f);
}
function Wf(e, t) {
  const n = window.getComputedStyle(e), r = (_) => (n[_] || "").split(", "), o = r(`${en}Delay`), i = r(`${en}Duration`), s = Fs(o, i), a = r(`${Xn}Delay`), l = r(`${Xn}Duration`), u = Fs(a, l);
  let c = null, d = 0, f = 0;
  t === en ? s > 0 && (c = en, d = s, f = i.length) : t === Xn ? u > 0 && (c = Xn, d = u, f = l.length) : (d = Math.max(s, u), c = d > 0 ? s > u ? en : Xn : null, f = c ? c === en ? i.length : l.length : 0);
  const h = c === en && /\b(transform|all)(,|$)/.test(
    r(`${en}Property`).toString()
  );
  return {
    type: c,
    timeout: d,
    propCount: f,
    hasTransform: h
  };
}
function Fs(e, t) {
  for (; e.length < t.length; )
    e = e.concat(e);
  return Math.max(...t.map((n, r) => js(n) + js(e[r])));
}
function js(e) {
  return e === "auto" ? 0 : Number(e.slice(0, -1).replace(",", ".")) * 1e3;
}
function Us() {
  return document.body.offsetHeight;
}
function Bf(e, t, n) {
  const r = e[Er];
  r && (t = (t ? [t, ...r] : [...r]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
const Vs = Symbol("_vod"), Kf = Symbol("_vsh"), Yf = Symbol(""), zf = /(^|;)\s*display\s*:/;
function Gf(e, t, n) {
  const r = e.style, o = Oe(n);
  let i = !1;
  if (n && !o) {
    if (t)
      if (Oe(t))
        for (const s of t.split(";")) {
          const a = s.slice(0, s.indexOf(":")).trim();
          n[a] == null && oo(r, a, "");
        }
      else
        for (const s in t)
          n[s] == null && oo(r, s, "");
    for (const s in n)
      s === "display" && (i = !0), oo(r, s, n[s]);
  } else if (o) {
    if (t !== n) {
      const s = r[Yf];
      s && (n += ";" + s), r.cssText = n, i = zf.test(n);
    }
  } else t && e.removeAttribute("style");
  Vs in e && (e[Vs] = i ? r.display : "", e[Kf] && (r.display = "none"));
}
const Hs = /\s*!important$/;
function oo(e, t, n) {
  if (X(n))
    n.forEach((r) => oo(e, t, r));
  else if (n == null && (n = ""), t.startsWith("--"))
    e.setProperty(t, n);
  else {
    const r = Xf(e, t);
    Hs.test(n) ? e.setProperty(
      wn(r),
      n.replace(Hs, ""),
      "important"
    ) : e[r] = n;
  }
}
const Ws = ["Webkit", "Moz", "ms"], zo = {};
function Xf(e, t) {
  const n = zo[t];
  if (n)
    return n;
  let r = pt(t);
  if (r !== "filter" && r in e)
    return zo[t] = r;
  r = So(r);
  for (let o = 0; o < Ws.length; o++) {
    const i = Ws[o] + r;
    if (i in e)
      return zo[t] = i;
  }
  return t;
}
const Bs = "http://www.w3.org/1999/xlink";
function Ks(e, t, n, r, o, i = Qc(t)) {
  r && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(Bs, t.slice(6, t.length)) : e.setAttributeNS(Bs, t, n) : n == null || i && !bl(n) ? e.removeAttribute(t) : e.setAttribute(
    t,
    i ? "" : qt(n) ? String(n) : n
  );
}
function Ys(e, t, n, r, o) {
  if (t === "innerHTML" || t === "textContent") {
    n != null && (e[t] = t === "innerHTML" ? ku(n) : n);
    return;
  }
  const i = e.tagName;
  if (t === "value" && i !== "PROGRESS" && // custom elements may use _value internally
  !i.includes("-")) {
    const a = i === "OPTION" ? e.getAttribute("value") || "" : e.value, l = n == null ? (
      // #11647: value should be set as empty string for null and undefined,
      // but <input type="checkbox"> should be set as 'on'.
      e.type === "checkbox" ? "on" : ""
    ) : String(n);
    (a !== l || !("_value" in e)) && (e.value = l), n == null && e.removeAttribute(t), e._value = n;
    return;
  }
  let s = !1;
  if (n === "" || n == null) {
    const a = typeof e[t];
    a === "boolean" ? n = bl(n) : n == null && a === "string" ? (n = "", s = !0) : a === "number" && (n = 0, s = !0);
  }
  try {
    e[t] = n;
  } catch {
  }
  s && e.removeAttribute(o || t);
}
function Jf(e, t, n, r) {
  e.addEventListener(t, n, r);
}
function qf(e, t, n, r) {
  e.removeEventListener(t, n, r);
}
const zs = Symbol("_vei");
function Zf(e, t, n, r, o = null) {
  const i = e[zs] || (e[zs] = {}), s = i[t];
  if (r && s)
    s.value = r;
  else {
    const [a, l] = Qf(t);
    if (r) {
      const u = i[t] = np(
        r,
        o
      );
      Jf(e, a, u, l);
    } else s && (qf(e, a, s, l), i[t] = void 0);
  }
}
const Gs = /(?:Once|Passive|Capture)$/;
function Qf(e) {
  let t;
  if (Gs.test(e)) {
    t = {};
    let r;
    for (; r = e.match(Gs); )
      e = e.slice(0, e.length - r[0].length), t[r[0].toLowerCase()] = !0;
  }
  return [e[2] === ":" ? e.slice(3) : wn(e.slice(2)), t];
}
let Go = 0;
const ep = /* @__PURE__ */ Promise.resolve(), tp = () => Go || (ep.then(() => Go = 0), Go = Date.now());
function np(e, t) {
  const n = (r) => {
    if (!r._vts)
      r._vts = Date.now();
    else if (r._vts <= n.attached)
      return;
    Ot(
      rp(r, n.value),
      t,
      5,
      [r]
    );
  };
  return n.value = e, n.attached = tp(), n;
}
function rp(e, t) {
  if (X(t)) {
    const n = e.stopImmediatePropagation;
    return e.stopImmediatePropagation = () => {
      n.call(e), e._stopped = !0;
    }, t.map(
      (r) => (o) => !o._stopped && r && r(o)
    );
  } else
    return t;
}
const Xs = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && // lowercase letter
e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, op = (e, t, n, r, o, i) => {
  const s = o === "svg";
  t === "class" ? Bf(e, r, s) : t === "style" ? Gf(e, n, r) : vo(t) ? $i(t) || Zf(e, t, n, r, i) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : ip(e, t, r, s)) ? (Ys(e, t, r), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && Ks(e, t, r, s, i, t !== "value")) : /* #11081 force set props for possible async custom element */ e._isVueCE && (/[A-Z]/.test(t) || !Oe(r)) ? Ys(e, pt(t), r, i, t) : (t === "true-value" ? e._trueValue = r : t === "false-value" && (e._falseValue = r), Ks(e, t, r, s));
};
function ip(e, t, n, r) {
  if (r)
    return !!(t === "innerHTML" || t === "textContent" || t in e && Xs(t) && q(n));
  if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA")
    return !1;
  if (t === "width" || t === "height") {
    const o = e.tagName;
    if (o === "IMG" || o === "VIDEO" || o === "CANVAS" || o === "SOURCE")
      return !1;
  }
  return Xs(t) && Oe(n) ? !1 : t in e;
}
const sp = /* @__PURE__ */ Ie({ patchProp: op }, Rf);
let Js;
function ap() {
  return Js || (Js = af(sp));
}
const lp = (...e) => {
  const t = ap().createApp(...e), { mount: n } = t;
  return t.mount = (r) => {
    const o = cp(r);
    if (!o) return;
    const i = t._component;
    !q(i) && !i.render && !i.template && (i.template = o.innerHTML), o.nodeType === 1 && (o.textContent = "");
    const s = n(o, !1, up(o));
    return o instanceof Element && (o.removeAttribute("v-cloak"), o.setAttribute("data-v-app", "")), s;
  }, t;
};
function up(e) {
  if (e instanceof SVGElement)
    return "svg";
  if (typeof MathMLElement == "function" && e instanceof MathMLElement)
    return "mathml";
}
function cp(e) {
  return Oe(e) ? document.querySelector(e) : e;
}
/*!
 * pinia v2.3.1
 * (c) 2025 Eduardo San Martin Morote
 * @license MIT
 */
const dp = (
  /* istanbul ignore next */
  Symbol()
);
var qs;
(function(e) {
  e.direct = "direct", e.patchObject = "patch object", e.patchFunction = "patch function";
})(qs || (qs = {}));
function fp() {
  const e = Sl(!0), t = e.run(() => De({}));
  let n = [], r = [];
  const o = Rl({
    install(i) {
      o._a = i, i.provide(dp, o), i.config.globalProperties.$pinia = o, r.forEach((s) => n.push(s)), r = [];
    },
    use(i) {
      return this._a ? n.push(i) : r.push(i), this;
    },
    _p: n,
    // it's actually undefined here
    // @ts-expect-error
    _a: null,
    _e: e,
    _s: /* @__PURE__ */ new Map(),
    state: t
  });
  return o;
}
var pp = Object.defineProperty, Zs = Object.getOwnPropertySymbols, mp = Object.prototype.hasOwnProperty, hp = Object.prototype.propertyIsEnumerable, Qs = (e, t, n) => t in e ? pp(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : e[t] = n, gp = (e, t) => {
  for (var n in t || (t = {})) mp.call(t, n) && Qs(e, n, t[n]);
  if (Zs) for (var n of Zs(t)) hp.call(t, n) && Qs(e, n, t[n]);
  return e;
};
function kn(e) {
  return e == null || e === "" || Array.isArray(e) && e.length === 0 || !(e instanceof Date) && typeof e == "object" && Object.keys(e).length === 0;
}
function es(e) {
  return typeof e == "function" && "call" in e && "apply" in e;
}
function _e(e) {
  return !kn(e);
}
function Rt(e, t = !0) {
  return e instanceof Object && e.constructor === Object && (t || Object.keys(e).length !== 0);
}
function Nu(e = {}, t = {}) {
  let n = gp({}, e);
  return Object.keys(t).forEach((r) => {
    let o = r;
    Rt(t[o]) && o in e && Rt(e[o]) ? n[o] = Nu(e[o], t[o]) : n[o] = t[o];
  }), n;
}
function bp(...e) {
  return e.reduce((t, n, r) => r === 0 ? n : Nu(t, n), {});
}
function lt(e, ...t) {
  return es(e) ? e(...t) : e;
}
function ot(e, t = !0) {
  return typeof e == "string" && (t || e !== "");
}
function At(e) {
  return ot(e) ? e.replace(/(-|_)/g, "").toLowerCase() : e;
}
function ts(e, t = "", n = {}) {
  let r = At(t).split("."), o = r.shift();
  if (o) {
    if (Rt(e)) {
      let i = Object.keys(e).find((s) => At(s) === o) || "";
      return ts(lt(e[i], n), r.join("."), n);
    }
    return;
  }
  return lt(e, n);
}
function Iu(e, t = !0) {
  return Array.isArray(e) && (t || e.length !== 0);
}
function vp(e) {
  return _e(e) && !isNaN(e);
}
function On(e, t) {
  if (t) {
    let n = t.test(e);
    return t.lastIndex = 0, n;
  }
  return !1;
}
function yp(...e) {
  return bp(...e);
}
function fr(e) {
  return e && e.replace(/\/\*(?:(?!\*\/)[\s\S])*\*\/|[\r\n\t]+/g, "").replace(/ {2,}/g, " ").replace(/ ([{:}]) /g, "$1").replace(/([;,]) /g, "$1").replace(/ !/g, "!").replace(/: /g, ":").trim();
}
function _p(e) {
  return ot(e, !1) ? e[0].toUpperCase() + e.slice(1) : e;
}
function Au(e) {
  return ot(e) ? e.replace(/(_)/g, "-").replace(/[A-Z]/g, (t, n) => n === 0 ? t : "-" + t.toLowerCase()).toLowerCase() : e;
}
function xu() {
  let e = /* @__PURE__ */ new Map();
  return { on(t, n) {
    let r = e.get(t);
    return r ? r.push(n) : r = [n], e.set(t, r), this;
  }, off(t, n) {
    let r = e.get(t);
    return r && r.splice(r.indexOf(n) >>> 0, 1), this;
  }, emit(t, n) {
    let r = e.get(t);
    r && r.forEach((o) => {
      o(n);
    });
  }, clear() {
    e.clear();
  } };
}
function Wn(...e) {
  if (e) {
    let t = [];
    for (let n = 0; n < e.length; n++) {
      let r = e[n];
      if (!r) continue;
      let o = typeof r;
      if (o === "string" || o === "number") t.push(r);
      else if (o === "object") {
        let i = Array.isArray(r) ? [Wn(...r)] : Object.entries(r).map(([s, a]) => a ? s : void 0);
        t = i.length ? t.concat(i.filter((s) => !!s)) : t;
      }
    }
    return t.join(" ").trim();
  }
}
function Sp(e, t) {
  return e ? e.classList ? e.classList.contains(t) : new RegExp("(^| )" + t + "( |$)", "gi").test(e.className) : !1;
}
function po(e, t) {
  if (e && t) {
    let n = (r) => {
      Sp(e, r) || (e.classList ? e.classList.add(r) : e.className += " " + r);
    };
    [t].flat().filter(Boolean).forEach((r) => r.split(" ").forEach(n));
  }
}
function Ep() {
  return window.innerWidth - document.documentElement.offsetWidth;
}
function Tp(e) {
  typeof e == "string" ? po(document.body, e || "p-overflow-hidden") : (e != null && e.variableName && document.body.style.setProperty(e.variableName, Ep() + "px"), po(document.body, (e == null ? void 0 : e.className) || "p-overflow-hidden"));
}
function pr(e, t) {
  if (e && t) {
    let n = (r) => {
      e.classList ? e.classList.remove(r) : e.className = e.className.replace(new RegExp("(^|\\b)" + r.split(" ").join("|") + "(\\b|$)", "gi"), " ");
    };
    [t].flat().filter(Boolean).forEach((r) => r.split(" ").forEach(n));
  }
}
function Cp(e) {
  typeof e == "string" ? pr(document.body, e || "p-overflow-hidden") : (e != null && e.variableName && document.body.style.removeProperty(e.variableName), pr(document.body, (e == null ? void 0 : e.className) || "p-overflow-hidden"));
}
function Op() {
  let e = window, t = document, n = t.documentElement, r = t.getElementsByTagName("body")[0], o = e.innerWidth || n.clientWidth || r.clientWidth, i = e.innerHeight || n.clientHeight || r.clientHeight;
  return { width: o, height: i };
}
function ea(e) {
  return e ? Math.abs(e.scrollLeft) : 0;
}
function Lp(e, t) {
  e && (typeof t == "string" ? e.style.cssText = t : Object.entries(t || {}).forEach(([n, r]) => e.style[n] = r));
}
function Du(e, t) {
  return e instanceof HTMLElement ? e.offsetWidth : 0;
}
function Pp(e) {
  if (e) {
    let t = e.parentNode;
    return t && t instanceof ShadowRoot && t.host && (t = t.host), t;
  }
  return null;
}
function wp(e) {
  return !!(e !== null && typeof e < "u" && e.nodeName && Pp(e));
}
function $n(e) {
  return typeof Element < "u" ? e instanceof Element : e !== null && typeof e == "object" && e.nodeType === 1 && typeof e.nodeName == "string";
}
function mo(e, t = {}) {
  if ($n(e)) {
    let n = (r, o) => {
      var i, s;
      let a = (i = e == null ? void 0 : e.$attrs) != null && i[r] ? [(s = e == null ? void 0 : e.$attrs) == null ? void 0 : s[r]] : [];
      return [o].flat().reduce((l, u) => {
        if (u != null) {
          let c = typeof u;
          if (c === "string" || c === "number") l.push(u);
          else if (c === "object") {
            let d = Array.isArray(u) ? n(r, u) : Object.entries(u).map(([f, h]) => r === "style" && (h || h === 0) ? `${f.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase()}:${h}` : h ? f : void 0);
            l = d.length ? l.concat(d.filter((f) => !!f)) : l;
          }
        }
        return l;
      }, a);
    };
    Object.entries(t).forEach(([r, o]) => {
      if (o != null) {
        let i = r.match(/^on(.+)/);
        i ? e.addEventListener(i[1].toLowerCase(), o) : r === "p-bind" || r === "pBind" ? mo(e, o) : (o = r === "class" ? [...new Set(n("class", o))].join(" ").trim() : r === "style" ? n("style", o).join(";").trim() : o, (e.$attrs = e.$attrs || {}) && (e.$attrs[r] = o), e.setAttribute(r, o));
      }
    });
  }
}
function Ru(e, t = {}, ...n) {
  {
    let r = document.createElement(e);
    return mo(r, t), r.append(...n), r;
  }
}
function kp(e, t) {
  return $n(e) ? Array.from(e.querySelectorAll(t)) : [];
}
function $p(e, t) {
  return $n(e) ? e.matches(t) ? e : e.querySelector(t) : null;
}
function Dn(e, t) {
  e && document.activeElement !== e && e.focus(t);
}
function Np(e, t) {
  if ($n(e)) {
    let n = e.getAttribute(t);
    return isNaN(n) ? n === "true" || n === "false" ? n === "true" : n : +n;
  }
}
function Mu(e, t = "") {
  let n = kp(e, `button:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [href]:not([tabindex = "-1"]):not([style*="display:none"]):not([hidden])${t},
            input:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            select:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            textarea:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [tabIndex]:not([tabIndex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [contenteditable]:not([tabIndex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t}`), r = [];
  for (let o of n) getComputedStyle(o).display != "none" && getComputedStyle(o).visibility != "hidden" && r.push(o);
  return r;
}
function Jn(e, t) {
  let n = Mu(e, t);
  return n.length > 0 ? n[0] : null;
}
function ta(e) {
  if (e) {
    let t = e.offsetHeight, n = getComputedStyle(e);
    return t -= parseFloat(n.paddingTop) + parseFloat(n.paddingBottom) + parseFloat(n.borderTopWidth) + parseFloat(n.borderBottomWidth), t;
  }
  return 0;
}
function Ip(e, t) {
  let n = Mu(e, t);
  return n.length > 0 ? n[n.length - 1] : null;
}
function Ap(e) {
  if (e) {
    let t = e.getBoundingClientRect();
    return { top: t.top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0), left: t.left + (window.pageXOffset || ea(document.documentElement) || ea(document.body) || 0) };
  }
  return { top: "auto", left: "auto" };
}
function Fu(e, t) {
  return e ? e.offsetHeight : 0;
}
function na(e) {
  if (e) {
    let t = e.offsetWidth, n = getComputedStyle(e);
    return t -= parseFloat(n.paddingLeft) + parseFloat(n.paddingRight) + parseFloat(n.borderLeftWidth) + parseFloat(n.borderRightWidth), t;
  }
  return 0;
}
function ju() {
  return !!(typeof window < "u" && window.document && window.document.createElement);
}
function ra(e, t = "") {
  return $n(e) ? e.matches(`button:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [href][clientHeight][clientWidth]:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            input:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            select:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            textarea:not([tabindex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [tabIndex]:not([tabIndex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t},
            [contenteditable]:not([tabIndex = "-1"]):not([disabled]):not([style*="display:none"]):not([hidden])${t}`) : !1;
}
function Uu(e, t = "", n) {
  $n(e) && n !== null && n !== void 0 && e.setAttribute(t, n);
}
var Xr = {};
function xp(e = "pui_id_") {
  return Object.hasOwn(Xr, e) || (Xr[e] = 0), Xr[e]++, `${e}${Xr[e]}`;
}
function Dp() {
  let e = [], t = (s, a, l = 999) => {
    let u = o(s, a, l), c = u.value + (u.key === s ? 0 : l) + 1;
    return e.push({ key: s, value: c }), c;
  }, n = (s) => {
    e = e.filter((a) => a.value !== s);
  }, r = (s, a) => o(s).value, o = (s, a, l = 0) => [...e].reverse().find((u) => !0) || { key: s, value: l }, i = (s) => s && parseInt(s.style.zIndex, 10) || 0;
  return { get: i, set: (s, a, l) => {
    a && (a.style.zIndex = String(t(s, !0, l)));
  }, clear: (s) => {
    s && (n(i(s)), s.style.zIndex = "");
  }, getCurrent: (s) => r(s) };
}
var Xo = Dp(), Rp = Object.defineProperty, Mp = Object.defineProperties, Fp = Object.getOwnPropertyDescriptors, ho = Object.getOwnPropertySymbols, Vu = Object.prototype.hasOwnProperty, Hu = Object.prototype.propertyIsEnumerable, oa = (e, t, n) => t in e ? Rp(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : e[t] = n, yt = (e, t) => {
  for (var n in t || (t = {})) Vu.call(t, n) && oa(e, n, t[n]);
  if (ho) for (var n of ho(t)) Hu.call(t, n) && oa(e, n, t[n]);
  return e;
}, Jo = (e, t) => Mp(e, Fp(t)), Ht = (e, t) => {
  var n = {};
  for (var r in e) Vu.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r]);
  if (e != null && ho) for (var r of ho(e)) t.indexOf(r) < 0 && Hu.call(e, r) && (n[r] = e[r]);
  return n;
}, jp = xu(), xe = jp, Tr = /{([^}]*)}/g, Wu = /(\d+\s+[\+\-\*\/]\s+\d+)/g, Bu = /var\([^)]+\)/g;
function ia(e) {
  return ot(e) ? e.replace(/[A-Z]/g, (t, n) => n === 0 ? t : "." + t.toLowerCase()).toLowerCase() : e;
}
function Up(e) {
  return Rt(e) && e.hasOwnProperty("$value") && e.hasOwnProperty("$type") ? e.$value : e;
}
function Vp(e) {
  return e.replaceAll(/ /g, "").replace(/[^\w]/g, "-");
}
function gi(e = "", t = "") {
  return Vp(`${ot(e, !1) && ot(t, !1) ? `${e}-` : e}${t}`);
}
function Ku(e = "", t = "") {
  return `--${gi(e, t)}`;
}
function Hp(e = "") {
  let t = (e.match(/{/g) || []).length, n = (e.match(/}/g) || []).length;
  return (t + n) % 2 !== 0;
}
function Yu(e, t = "", n = "", r = [], o) {
  if (ot(e)) {
    let i = e.trim();
    if (Hp(i)) return;
    if (On(i, Tr)) {
      let s = i.replaceAll(Tr, (a) => {
        let l = a.replace(/{|}/g, "").split(".").filter((u) => !r.some((c) => On(u, c)));
        return `var(${Ku(n, Au(l.join("-")))}${_e(o) ? `, ${o}` : ""})`;
      });
      return On(s.replace(Bu, "0"), Wu) ? `calc(${s})` : s;
    }
    return i;
  } else if (vp(e)) return e;
}
function Wp(e, t, n) {
  ot(t, !1) && e.push(`${t}:${n};`);
}
function An(e, t) {
  return e ? `${e}{${t}}` : "";
}
function zu(e, t) {
  if (e.indexOf("dt(") === -1) return e;
  function n(s, a) {
    let l = [], u = 0, c = "", d = null, f = 0;
    for (; u <= s.length; ) {
      let h = s[u];
      if ((h === '"' || h === "'" || h === "`") && s[u - 1] !== "\\" && (d = d === h ? null : h), !d && (h === "(" && f++, h === ")" && f--, (h === "," || u === s.length) && f === 0)) {
        let _ = c.trim();
        _.startsWith("dt(") ? l.push(zu(_, a)) : l.push(r(_)), c = "", u++;
        continue;
      }
      h !== void 0 && (c += h), u++;
    }
    return l;
  }
  function r(s) {
    let a = s[0];
    if ((a === '"' || a === "'" || a === "`") && s[s.length - 1] === a) return s.slice(1, -1);
    let l = Number(s);
    return isNaN(l) ? s : l;
  }
  let o = [], i = [];
  for (let s = 0; s < e.length; s++) if (e[s] === "d" && e.slice(s, s + 3) === "dt(") i.push(s), s += 2;
  else if (e[s] === ")" && i.length > 0) {
    let a = i.pop();
    i.length === 0 && o.push([a, s]);
  }
  if (!o.length) return e;
  for (let s = o.length - 1; s >= 0; s--) {
    let [a, l] = o[s], u = e.slice(a + 3, l), c = n(u, t), d = t(...c);
    e = e.slice(0, a) + d + e.slice(l + 1);
  }
  return e;
}
var Gu = (e) => {
  var t;
  let n = fe.getTheme(), r = bi(n, e, void 0, "variable"), o = (t = r == null ? void 0 : r.match(/--[\w-]+/g)) == null ? void 0 : t[0], i = bi(n, e, void 0, "value");
  return { name: o, variable: r, value: i };
}, Ln = (...e) => bi(fe.getTheme(), ...e), bi = (e = {}, t, n, r) => {
  if (t) {
    let { variable: o, options: i } = fe.defaults || {}, { prefix: s, transform: a } = (e == null ? void 0 : e.options) || i || {}, l = On(t, Tr) ? t : `{${t}}`;
    return r === "value" || kn(r) && a === "strict" ? fe.getTokenValue(t) : Yu(l, void 0, s, [o.excludedKeyRegex], n);
  }
  return "";
};
function Jr(e, ...t) {
  if (e instanceof Array) {
    let n = e.reduce((r, o, i) => {
      var s;
      return r + o + ((s = lt(t[i], { dt: Ln })) != null ? s : "");
    }, "");
    return zu(n, Ln);
  }
  return lt(e, { dt: Ln });
}
function Bp(e, t = {}) {
  let n = fe.defaults.variable, { prefix: r = n.prefix, selector: o = n.selector, excludedKeyRegex: i = n.excludedKeyRegex } = t, s = [], a = [], l = [{ node: e, path: r }];
  for (; l.length; ) {
    let { node: c, path: d } = l.pop();
    for (let f in c) {
      let h = c[f], _ = Up(h), E = On(f, i) ? gi(d) : gi(d, Au(f));
      if (Rt(_)) l.push({ node: _, path: E });
      else {
        let w = Ku(E), P = Yu(_, E, r, [i]);
        Wp(a, w, P);
        let M = E;
        r && M.startsWith(r + "-") && (M = M.slice(r.length + 1)), s.push(M.replace(/-/g, "."));
      }
    }
  }
  let u = a.join("");
  return { value: a, tokens: s, declarations: u, css: An(o, u) };
}
var gt = { regex: { rules: { class: { pattern: /^\.([a-zA-Z][\w-]*)$/, resolve(e) {
  return { type: "class", selector: e, matched: this.pattern.test(e.trim()) };
} }, attr: { pattern: /^\[(.*)\]$/, resolve(e) {
  return { type: "attr", selector: `:root${e},:host${e}`, matched: this.pattern.test(e.trim()) };
} }, media: { pattern: /^@media (.*)$/, resolve(e) {
  return { type: "media", selector: e, matched: this.pattern.test(e.trim()) };
} }, system: { pattern: /^system$/, resolve(e) {
  return { type: "system", selector: "@media (prefers-color-scheme: dark)", matched: this.pattern.test(e.trim()) };
} }, custom: { resolve(e) {
  return { type: "custom", selector: e, matched: !0 };
} } }, resolve(e) {
  let t = Object.keys(this.rules).filter((n) => n !== "custom").map((n) => this.rules[n]);
  return [e].flat().map((n) => {
    var r;
    return (r = t.map((o) => o.resolve(n)).find((o) => o.matched)) != null ? r : this.rules.custom.resolve(n);
  });
} }, _toVariables(e, t) {
  return Bp(e, { prefix: t == null ? void 0 : t.prefix });
}, getCommon({ name: e = "", theme: t = {}, params: n, set: r, defaults: o }) {
  var i, s, a, l, u, c, d;
  let { preset: f, options: h } = t, _, E, w, P, M, S, g;
  if (_e(f) && h.transform !== "strict") {
    let { primitive: O, semantic: L, extend: A } = f, F = L || {}, { colorScheme: $ } = F, B = Ht(F, ["colorScheme"]), Y = A || {}, { colorScheme: R } = Y, z = Ht(Y, ["colorScheme"]), ae = $ || {}, { dark: Te } = ae, ne = Ht(ae, ["dark"]), te = R || {}, { dark: Q } = te, Pe = Ht(te, ["dark"]), we = _e(O) ? this._toVariables({ primitive: O }, h) : {}, ue = _e(B) ? this._toVariables({ semantic: B }, h) : {}, he = _e(ne) ? this._toVariables({ light: ne }, h) : {}, it = _e(Te) ? this._toVariables({ dark: Te }, h) : {}, Be = _e(z) ? this._toVariables({ semantic: z }, h) : {}, Lt = _e(Pe) ? this._toVariables({ light: Pe }, h) : {}, Qe = _e(Q) ? this._toVariables({ dark: Q }, h) : {}, [mt, ht] = [(i = we.declarations) != null ? i : "", we.tokens], [Pt, et] = [(s = ue.declarations) != null ? s : "", ue.tokens || []], [Qt, b] = [(a = he.declarations) != null ? a : "", he.tokens || []], [y, v] = [(l = it.declarations) != null ? l : "", it.tokens || []], [N, x] = [(u = Be.declarations) != null ? u : "", Be.tokens || []], [D, V] = [(c = Lt.declarations) != null ? c : "", Lt.tokens || []], [U, p] = [(d = Qe.declarations) != null ? d : "", Qe.tokens || []];
    _ = this.transformCSS(e, mt, "light", "variable", h, r, o), E = ht;
    let m = this.transformCSS(e, `${Pt}${Qt}`, "light", "variable", h, r, o), T = this.transformCSS(e, `${y}`, "dark", "variable", h, r, o);
    w = `${m}${T}`, P = [.../* @__PURE__ */ new Set([...et, ...b, ...v])];
    let k = this.transformCSS(e, `${N}${D}color-scheme:light`, "light", "variable", h, r, o), H = this.transformCSS(e, `${U}color-scheme:dark`, "dark", "variable", h, r, o);
    M = `${k}${H}`, S = [.../* @__PURE__ */ new Set([...x, ...V, ...p])], g = lt(f.css, { dt: Ln });
  }
  return { primitive: { css: _, tokens: E }, semantic: { css: w, tokens: P }, global: { css: M, tokens: S }, style: g };
}, getPreset({ name: e = "", preset: t = {}, options: n, params: r, set: o, defaults: i, selector: s }) {
  var a, l, u;
  let c, d, f;
  if (_e(t) && n.transform !== "strict") {
    let h = e.replace("-directive", ""), _ = t, { colorScheme: E, extend: w, css: P } = _, M = Ht(_, ["colorScheme", "extend", "css"]), S = w || {}, { colorScheme: g } = S, O = Ht(S, ["colorScheme"]), L = E || {}, { dark: A } = L, F = Ht(L, ["dark"]), $ = g || {}, { dark: B } = $, Y = Ht($, ["dark"]), R = _e(M) ? this._toVariables({ [h]: yt(yt({}, M), O) }, n) : {}, z = _e(F) ? this._toVariables({ [h]: yt(yt({}, F), Y) }, n) : {}, ae = _e(A) ? this._toVariables({ [h]: yt(yt({}, A), B) }, n) : {}, [Te, ne] = [(a = R.declarations) != null ? a : "", R.tokens || []], [te, Q] = [(l = z.declarations) != null ? l : "", z.tokens || []], [Pe, we] = [(u = ae.declarations) != null ? u : "", ae.tokens || []], ue = this.transformCSS(h, `${Te}${te}`, "light", "variable", n, o, i, s), he = this.transformCSS(h, Pe, "dark", "variable", n, o, i, s);
    c = `${ue}${he}`, d = [.../* @__PURE__ */ new Set([...ne, ...Q, ...we])], f = lt(P, { dt: Ln });
  }
  return { css: c, tokens: d, style: f };
}, getPresetC({ name: e = "", theme: t = {}, params: n, set: r, defaults: o }) {
  var i;
  let { preset: s, options: a } = t, l = (i = s == null ? void 0 : s.components) == null ? void 0 : i[e];
  return this.getPreset({ name: e, preset: l, options: a, params: n, set: r, defaults: o });
}, getPresetD({ name: e = "", theme: t = {}, params: n, set: r, defaults: o }) {
  var i, s;
  let a = e.replace("-directive", ""), { preset: l, options: u } = t, c = ((i = l == null ? void 0 : l.components) == null ? void 0 : i[a]) || ((s = l == null ? void 0 : l.directives) == null ? void 0 : s[a]);
  return this.getPreset({ name: a, preset: c, options: u, params: n, set: r, defaults: o });
}, applyDarkColorScheme(e) {
  return !(e.darkModeSelector === "none" || e.darkModeSelector === !1);
}, getColorSchemeOption(e, t) {
  var n;
  return this.applyDarkColorScheme(e) ? this.regex.resolve(e.darkModeSelector === !0 ? t.options.darkModeSelector : (n = e.darkModeSelector) != null ? n : t.options.darkModeSelector) : [];
}, getLayerOrder(e, t = {}, n, r) {
  let { cssLayer: o } = t;
  return o ? `@layer ${lt(o.order || o.name || "primeui", n)}` : "";
}, getCommonStyleSheet({ name: e = "", theme: t = {}, params: n, props: r = {}, set: o, defaults: i }) {
  let s = this.getCommon({ name: e, theme: t, params: n, set: o, defaults: i }), a = Object.entries(r).reduce((l, [u, c]) => l.push(`${u}="${c}"`) && l, []).join(" ");
  return Object.entries(s || {}).reduce((l, [u, c]) => {
    if (Rt(c) && Object.hasOwn(c, "css")) {
      let d = fr(c.css), f = `${u}-variables`;
      l.push(`<style type="text/css" data-primevue-style-id="${f}" ${a}>${d}</style>`);
    }
    return l;
  }, []).join("");
}, getStyleSheet({ name: e = "", theme: t = {}, params: n, props: r = {}, set: o, defaults: i }) {
  var s;
  let a = { name: e, theme: t, params: n, set: o, defaults: i }, l = (s = e.includes("-directive") ? this.getPresetD(a) : this.getPresetC(a)) == null ? void 0 : s.css, u = Object.entries(r).reduce((c, [d, f]) => c.push(`${d}="${f}"`) && c, []).join(" ");
  return l ? `<style type="text/css" data-primevue-style-id="${e}-variables" ${u}>${fr(l)}</style>` : "";
}, createTokens(e = {}, t, n = "", r = "", o = {}) {
  let i = function(a, l = {}, u = []) {
    if (u.includes(this.path)) return console.warn(`Circular reference detected at ${this.path}`), { colorScheme: a, path: this.path, paths: l, value: void 0 };
    u.push(this.path), l.name = this.path, l.binding || (l.binding = {});
    let c = this.value;
    if (typeof this.value == "string" && Tr.test(this.value)) {
      let d = this.value.trim().replace(Tr, (f) => {
        var h;
        let _ = f.slice(1, -1), E = this.tokens[_];
        if (!E) return console.warn(`Token not found for path: ${_}`), "__UNRESOLVED__";
        let w = E.computed(a, l, u);
        return Array.isArray(w) && w.length === 2 ? `light-dark(${w[0].value},${w[1].value})` : (h = w == null ? void 0 : w.value) != null ? h : "__UNRESOLVED__";
      });
      c = Wu.test(d.replace(Bu, "0")) ? `calc(${d})` : d;
    }
    return kn(l.binding) && delete l.binding, u.pop(), { colorScheme: a, path: this.path, paths: l, value: c.includes("__UNRESOLVED__") ? void 0 : c };
  }, s = (a, l, u) => {
    Object.entries(a).forEach(([c, d]) => {
      let f = On(c, t.variable.excludedKeyRegex) ? l : l ? `${l}.${ia(c)}` : ia(c), h = u ? `${u}.${c}` : c;
      Rt(d) ? s(d, f, h) : (o[f] || (o[f] = { paths: [], computed: (_, E = {}, w = []) => {
        if (o[f].paths.length === 1) return o[f].paths[0].computed(o[f].paths[0].scheme, E.binding, w);
        if (_ && _ !== "none") for (let P = 0; P < o[f].paths.length; P++) {
          let M = o[f].paths[P];
          if (M.scheme === _) return M.computed(_, E.binding, w);
        }
        return o[f].paths.map((P) => P.computed(P.scheme, E[P.scheme], w));
      } }), o[f].paths.push({ path: h, value: d, scheme: h.includes("colorScheme.light") ? "light" : h.includes("colorScheme.dark") ? "dark" : "none", computed: i, tokens: o }));
    });
  };
  return s(e, n, r), o;
}, getTokenValue(e, t, n) {
  var r;
  let o = ((a) => a.split(".").filter((l) => !On(l.toLowerCase(), n.variable.excludedKeyRegex)).join("."))(t), i = t.includes("colorScheme.light") ? "light" : t.includes("colorScheme.dark") ? "dark" : void 0, s = [(r = e[o]) == null ? void 0 : r.computed(i)].flat().filter((a) => a);
  return s.length === 1 ? s[0].value : s.reduce((a = {}, l) => {
    let u = l, { colorScheme: c } = u, d = Ht(u, ["colorScheme"]);
    return a[c] = d, a;
  }, void 0);
}, getSelectorRule(e, t, n, r) {
  return n === "class" || n === "attr" ? An(_e(t) ? `${e}${t},${e} ${t}` : e, r) : An(e, An(t ?? ":root,:host", r));
}, transformCSS(e, t, n, r, o = {}, i, s, a) {
  if (_e(t)) {
    let { cssLayer: l } = o;
    if (r !== "style") {
      let u = this.getColorSchemeOption(o, s);
      t = n === "dark" ? u.reduce((c, { type: d, selector: f }) => (_e(f) && (c += f.includes("[CSS]") ? f.replace("[CSS]", t) : this.getSelectorRule(f, a, d, t)), c), "") : An(a ?? ":root,:host", t);
    }
    if (l) {
      let u = { name: "primeui" };
      Rt(l) && (u.name = lt(l.name, { name: e, type: r })), _e(u.name) && (t = An(`@layer ${u.name}`, t), i == null || i.layerNames(u.name));
    }
    return t;
  }
  return "";
} }, fe = { defaults: { variable: { prefix: "p", selector: ":root,:host", excludedKeyRegex: /^(primitive|semantic|components|directives|variables|colorscheme|light|dark|common|root|states|extend|css)$/gi }, options: { prefix: "p", darkModeSelector: "system", cssLayer: !1 } }, _theme: void 0, _layerNames: /* @__PURE__ */ new Set(), _loadedStyleNames: /* @__PURE__ */ new Set(), _loadingStyles: /* @__PURE__ */ new Set(), _tokens: {}, update(e = {}) {
  let { theme: t } = e;
  t && (this._theme = Jo(yt({}, t), { options: yt(yt({}, this.defaults.options), t.options) }), this._tokens = gt.createTokens(this.preset, this.defaults), this.clearLoadedStyleNames());
}, get theme() {
  return this._theme;
}, get preset() {
  var e;
  return ((e = this.theme) == null ? void 0 : e.preset) || {};
}, get options() {
  var e;
  return ((e = this.theme) == null ? void 0 : e.options) || {};
}, get tokens() {
  return this._tokens;
}, getTheme() {
  return this.theme;
}, setTheme(e) {
  this.update({ theme: e }), xe.emit("theme:change", e);
}, getPreset() {
  return this.preset;
}, setPreset(e) {
  this._theme = Jo(yt({}, this.theme), { preset: e }), this._tokens = gt.createTokens(e, this.defaults), this.clearLoadedStyleNames(), xe.emit("preset:change", e), xe.emit("theme:change", this.theme);
}, getOptions() {
  return this.options;
}, setOptions(e) {
  this._theme = Jo(yt({}, this.theme), { options: e }), this.clearLoadedStyleNames(), xe.emit("options:change", e), xe.emit("theme:change", this.theme);
}, getLayerNames() {
  return [...this._layerNames];
}, setLayerNames(e) {
  this._layerNames.add(e);
}, getLoadedStyleNames() {
  return this._loadedStyleNames;
}, isStyleNameLoaded(e) {
  return this._loadedStyleNames.has(e);
}, setLoadedStyleName(e) {
  this._loadedStyleNames.add(e);
}, deleteLoadedStyleName(e) {
  this._loadedStyleNames.delete(e);
}, clearLoadedStyleNames() {
  this._loadedStyleNames.clear();
}, getTokenValue(e) {
  return gt.getTokenValue(this.tokens, e, this.defaults);
}, getCommon(e = "", t) {
  return gt.getCommon({ name: e, theme: this.theme, params: t, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } });
}, getComponent(e = "", t) {
  let n = { name: e, theme: this.theme, params: t, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } };
  return gt.getPresetC(n);
}, getDirective(e = "", t) {
  let n = { name: e, theme: this.theme, params: t, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } };
  return gt.getPresetD(n);
}, getCustomPreset(e = "", t, n, r) {
  let o = { name: e, preset: t, options: this.options, selector: n, params: r, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } };
  return gt.getPreset(o);
}, getLayerOrderCSS(e = "") {
  return gt.getLayerOrder(e, this.options, { names: this.getLayerNames() }, this.defaults);
}, transformCSS(e = "", t, n = "style", r) {
  return gt.transformCSS(e, t, r, n, this.options, { layerNames: this.setLayerNames.bind(this) }, this.defaults);
}, getCommonStyleSheet(e = "", t, n = {}) {
  return gt.getCommonStyleSheet({ name: e, theme: this.theme, params: t, props: n, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } });
}, getStyleSheet(e, t, n = {}) {
  return gt.getStyleSheet({ name: e, theme: this.theme, params: t, props: n, defaults: this.defaults, set: { layerNames: this.setLayerNames.bind(this) } });
}, onStyleMounted(e) {
  this._loadingStyles.add(e);
}, onStyleUpdated(e) {
  this._loadingStyles.add(e);
}, onStyleLoaded(e, { name: t }) {
  this._loadingStyles.size && (this._loadingStyles.delete(t), xe.emit(`theme:${t}:load`, e), !this._loadingStyles.size && xe.emit("theme:load"));
} }, je = {
  STARTS_WITH: "startsWith",
  CONTAINS: "contains",
  NOT_CONTAINS: "notContains",
  ENDS_WITH: "endsWith",
  EQUALS: "equals",
  NOT_EQUALS: "notEquals",
  LESS_THAN: "lt",
  LESS_THAN_OR_EQUAL_TO: "lte",
  GREATER_THAN: "gt",
  GREATER_THAN_OR_EQUAL_TO: "gte",
  DATE_IS: "dateIs",
  DATE_IS_NOT: "dateIsNot",
  DATE_BEFORE: "dateBefore",
  DATE_AFTER: "dateAfter"
}, Kp = `
    *,
    ::before,
    ::after {
        box-sizing: border-box;
    }

    /* Non vue overlay animations */
    .p-connected-overlay {
        opacity: 0;
        transform: scaleY(0.8);
        transition:
            transform 0.12s cubic-bezier(0, 0, 0.2, 1),
            opacity 0.12s cubic-bezier(0, 0, 0.2, 1);
    }

    .p-connected-overlay-visible {
        opacity: 1;
        transform: scaleY(1);
    }

    .p-connected-overlay-hidden {
        opacity: 0;
        transform: scaleY(1);
        transition: opacity 0.1s linear;
    }

    /* Vue based overlay animations */
    .p-connected-overlay-enter-from {
        opacity: 0;
        transform: scaleY(0.8);
    }

    .p-connected-overlay-leave-to {
        opacity: 0;
    }

    .p-connected-overlay-enter-active {
        transition:
            transform 0.12s cubic-bezier(0, 0, 0.2, 1),
            opacity 0.12s cubic-bezier(0, 0, 0.2, 1);
    }

    .p-connected-overlay-leave-active {
        transition: opacity 0.1s linear;
    }

    /* Toggleable Content */
    .p-toggleable-content-enter-from,
    .p-toggleable-content-leave-to {
        max-height: 0;
    }

    .p-toggleable-content-enter-to,
    .p-toggleable-content-leave-from {
        max-height: 1000px;
    }

    .p-toggleable-content-leave-active {
        overflow: hidden;
        transition: max-height 0.45s cubic-bezier(0, 1, 0, 1);
    }

    .p-toggleable-content-enter-active {
        overflow: hidden;
        transition: max-height 1s ease-in-out;
    }

    .p-disabled,
    .p-disabled * {
        cursor: default;
        pointer-events: none;
        user-select: none;
    }

    .p-disabled,
    .p-component:disabled {
        opacity: dt('disabled.opacity');
    }

    .pi {
        font-size: dt('icon.size');
    }

    .p-icon {
        width: dt('icon.size');
        height: dt('icon.size');
    }

    .p-overlay-mask {
        background: dt('mask.background');
        color: dt('mask.color');
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }

    .p-overlay-mask-enter {
        animation: p-overlay-mask-enter-animation dt('mask.transition.duration') forwards;
    }

    .p-overlay-mask-leave {
        animation: p-overlay-mask-leave-animation dt('mask.transition.duration') forwards;
    }

    @keyframes p-overlay-mask-enter-animation {
        from {
            background: transparent;
        }
        to {
            background: dt('mask.background');
        }
    }
    @keyframes p-overlay-mask-leave-animation {
        from {
            background: dt('mask.background');
        }
        to {
            background: transparent;
        }
    }
`;
function Cr(e) {
  "@babel/helpers - typeof";
  return Cr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Cr(e);
}
function sa(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function aa(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? sa(Object(n), !0).forEach(function(r) {
      Yp(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : sa(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function Yp(e, t, n) {
  return (t = zp(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function zp(e) {
  var t = Gp(e, "string");
  return Cr(t) == "symbol" ? t : t + "";
}
function Gp(e, t) {
  if (Cr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Cr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function Xp(e) {
  var t = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !0;
  jt() && jt().components ? Vr(e) : t ? e() : Hl(e);
}
var Jp = 0;
function qp(e) {
  var t = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, n = De(!1), r = De(e), o = De(null), i = ju() ? window.document : void 0, s = t.document, a = s === void 0 ? i : s, l = t.immediate, u = l === void 0 ? !0 : l, c = t.manual, d = c === void 0 ? !1 : c, f = t.name, h = f === void 0 ? "style_".concat(++Jp) : f, _ = t.id, E = _ === void 0 ? void 0 : _, w = t.media, P = w === void 0 ? void 0 : w, M = t.nonce, S = M === void 0 ? void 0 : M, g = t.first, O = g === void 0 ? !1 : g, L = t.onMounted, A = L === void 0 ? void 0 : L, F = t.onUpdated, $ = F === void 0 ? void 0 : F, B = t.onLoad, Y = B === void 0 ? void 0 : B, R = t.props, z = R === void 0 ? {} : R, ae = function() {
  }, Te = function(Q) {
    var Pe = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    if (a) {
      var we = aa(aa({}, z), Pe), ue = we.name || h, he = we.id || E, it = we.nonce || S;
      o.value = a.querySelector('style[data-primevue-style-id="'.concat(ue, '"]')) || a.getElementById(he) || a.createElement("style"), o.value.isConnected || (r.value = Q || e, mo(o.value, {
        type: "text/css",
        id: he,
        media: P,
        nonce: it
      }), O ? a.head.prepend(o.value) : a.head.appendChild(o.value), Uu(o.value, "data-primevue-style-id", ue), mo(o.value, we), o.value.onload = function(Be) {
        return Y == null ? void 0 : Y(Be, {
          name: ue
        });
      }, A == null || A(ue)), !n.value && (ae = ft(r, function(Be) {
        o.value.textContent = Be, $ == null || $(ue);
      }, {
        immediate: !0
      }), n.value = !0);
    }
  }, ne = function() {
    !a || !n.value || (ae(), wp(o.value) && a.head.removeChild(o.value), n.value = !1, o.value = null);
  };
  return u && !d && Xp(Te), {
    id: E,
    name: h,
    el: o,
    css: r,
    unload: ne,
    load: Te,
    isLoaded: Ui(n)
  };
}
function Or(e) {
  "@babel/helpers - typeof";
  return Or = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Or(e);
}
var la, ua, ca, da;
function fa(e, t) {
  return tm(e) || em(e, t) || Qp(e, t) || Zp();
}
function Zp() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Qp(e, t) {
  if (e) {
    if (typeof e == "string") return pa(e, t);
    var n = {}.toString.call(e).slice(8, -1);
    return n === "Object" && e.constructor && (n = e.constructor.name), n === "Map" || n === "Set" ? Array.from(e) : n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? pa(e, t) : void 0;
  }
}
function pa(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var n = 0, r = Array(t); n < t; n++) r[n] = e[n];
  return r;
}
function em(e, t) {
  var n = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (n != null) {
    var r, o, i, s, a = [], l = !0, u = !1;
    try {
      if (i = (n = n.call(e)).next, t !== 0) for (; !(l = (r = i.call(n)).done) && (a.push(r.value), a.length !== t); l = !0) ;
    } catch (c) {
      u = !0, o = c;
    } finally {
      try {
        if (!l && n.return != null && (s = n.return(), Object(s) !== s)) return;
      } finally {
        if (u) throw o;
      }
    }
    return a;
  }
}
function tm(e) {
  if (Array.isArray(e)) return e;
}
function ma(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function qo(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ma(Object(n), !0).forEach(function(r) {
      nm(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : ma(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function nm(e, t, n) {
  return (t = rm(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function rm(e) {
  var t = om(e, "string");
  return Or(t) == "symbol" ? t : t + "";
}
function om(e, t) {
  if (Or(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Or(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function qr(e, t) {
  return t || (t = e.slice(0)), Object.freeze(Object.defineProperties(e, { raw: { value: Object.freeze(t) } }));
}
var im = function(t) {
  var n = t.dt;
  return `
.p-hidden-accessible {
    border: 0;
    clip: rect(0 0 0 0);
    height: 1px;
    margin: -1px;
    opacity: 0;
    overflow: hidden;
    padding: 0;
    pointer-events: none;
    position: absolute;
    white-space: nowrap;
    width: 1px;
}

.p-overflow-hidden {
    overflow: hidden;
    padding-right: `.concat(n("scrollbar.width"), `;
}
`);
}, sm = {}, am = {}, ye = {
  name: "base",
  css: im,
  style: Kp,
  classes: sm,
  inlineStyles: am,
  load: function(t) {
    var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : function(i) {
      return i;
    }, o = r(Jr(la || (la = qr(["", ""])), t));
    return _e(o) ? qp(fr(o), qo({
      name: this.name
    }, n)) : {};
  },
  loadCSS: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    return this.load(this.css, t);
  },
  loadStyle: function() {
    var t = this, n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "";
    return this.load(this.style, n, function() {
      var o = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "";
      return fe.transformCSS(n.name || t.name, "".concat(o).concat(Jr(ua || (ua = qr(["", ""])), r)));
    });
  },
  getCommonTheme: function(t) {
    return fe.getCommon(this.name, t);
  },
  getComponentTheme: function(t) {
    return fe.getComponent(this.name, t);
  },
  getDirectiveTheme: function(t) {
    return fe.getDirective(this.name, t);
  },
  getPresetTheme: function(t, n, r) {
    return fe.getCustomPreset(this.name, t, n, r);
  },
  getLayerOrderThemeCSS: function() {
    return fe.getLayerOrderCSS(this.name);
  },
  getStyleSheet: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    if (this.css) {
      var r = lt(this.css, {
        dt: Ln
      }) || "", o = fr(Jr(ca || (ca = qr(["", "", ""])), r, t)), i = Object.entries(n).reduce(function(s, a) {
        var l = fa(a, 2), u = l[0], c = l[1];
        return s.push("".concat(u, '="').concat(c, '"')) && s;
      }, []).join(" ");
      return _e(o) ? '<style type="text/css" data-primevue-style-id="'.concat(this.name, '" ').concat(i, ">").concat(o, "</style>") : "";
    }
    return "";
  },
  getCommonThemeStyleSheet: function(t) {
    var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    return fe.getCommonStyleSheet(this.name, t, n);
  },
  getThemeStyleSheet: function(t) {
    var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, r = [fe.getStyleSheet(this.name, t, n)];
    if (this.style) {
      var o = this.name === "base" ? "global-style" : "".concat(this.name, "-style"), i = Jr(da || (da = qr(["", ""])), lt(this.style, {
        dt: Ln
      })), s = fr(fe.transformCSS(o, i)), a = Object.entries(n).reduce(function(l, u) {
        var c = fa(u, 2), d = c[0], f = c[1];
        return l.push("".concat(d, '="').concat(f, '"')) && l;
      }, []).join(" ");
      _e(s) && r.push('<style type="text/css" data-primevue-style-id="'.concat(o, '" ').concat(a, ">").concat(s, "</style>"));
    }
    return r.join("");
  },
  extend: function(t) {
    return qo(qo({}, this), {}, {
      css: void 0,
      style: void 0
    }, t);
  }
}, an = xu();
function Lr(e) {
  "@babel/helpers - typeof";
  return Lr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Lr(e);
}
function ha(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function Zr(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ha(Object(n), !0).forEach(function(r) {
      lm(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : ha(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function lm(e, t, n) {
  return (t = um(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function um(e) {
  var t = cm(e, "string");
  return Lr(t) == "symbol" ? t : t + "";
}
function cm(e, t) {
  if (Lr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Lr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var dm = {
  ripple: !1,
  inputStyle: null,
  inputVariant: null,
  locale: {
    startsWith: "Starts with",
    contains: "Contains",
    notContains: "Not contains",
    endsWith: "Ends with",
    equals: "Equals",
    notEquals: "Not equals",
    noFilter: "No Filter",
    lt: "Less than",
    lte: "Less than or equal to",
    gt: "Greater than",
    gte: "Greater than or equal to",
    dateIs: "Date is",
    dateIsNot: "Date is not",
    dateBefore: "Date is before",
    dateAfter: "Date is after",
    clear: "Clear",
    apply: "Apply",
    matchAll: "Match All",
    matchAny: "Match Any",
    addRule: "Add Rule",
    removeRule: "Remove Rule",
    accept: "Yes",
    reject: "No",
    choose: "Choose",
    upload: "Upload",
    cancel: "Cancel",
    completed: "Completed",
    pending: "Pending",
    fileSizeTypes: ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"],
    dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    dayNamesShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    dayNamesMin: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
    monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    monthNamesShort: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    chooseYear: "Choose Year",
    chooseMonth: "Choose Month",
    chooseDate: "Choose Date",
    prevDecade: "Previous Decade",
    nextDecade: "Next Decade",
    prevYear: "Previous Year",
    nextYear: "Next Year",
    prevMonth: "Previous Month",
    nextMonth: "Next Month",
    prevHour: "Previous Hour",
    nextHour: "Next Hour",
    prevMinute: "Previous Minute",
    nextMinute: "Next Minute",
    prevSecond: "Previous Second",
    nextSecond: "Next Second",
    am: "am",
    pm: "pm",
    today: "Today",
    weekHeader: "Wk",
    firstDayOfWeek: 0,
    showMonthAfterYear: !1,
    dateFormat: "mm/dd/yy",
    weak: "Weak",
    medium: "Medium",
    strong: "Strong",
    passwordPrompt: "Enter a password",
    emptyFilterMessage: "No results found",
    searchMessage: "{0} results are available",
    selectionMessage: "{0} items selected",
    emptySelectionMessage: "No selected item",
    emptySearchMessage: "No results found",
    fileChosenMessage: "{0} files",
    noFileChosenMessage: "No file chosen",
    emptyMessage: "No available options",
    aria: {
      trueLabel: "True",
      falseLabel: "False",
      nullLabel: "Not Selected",
      star: "1 star",
      stars: "{star} stars",
      selectAll: "All items selected",
      unselectAll: "All items unselected",
      close: "Close",
      previous: "Previous",
      next: "Next",
      navigation: "Navigation",
      scrollTop: "Scroll Top",
      moveTop: "Move Top",
      moveUp: "Move Up",
      moveDown: "Move Down",
      moveBottom: "Move Bottom",
      moveToTarget: "Move to Target",
      moveToSource: "Move to Source",
      moveAllToTarget: "Move All to Target",
      moveAllToSource: "Move All to Source",
      pageLabel: "Page {page}",
      firstPageLabel: "First Page",
      lastPageLabel: "Last Page",
      nextPageLabel: "Next Page",
      prevPageLabel: "Previous Page",
      rowsPerPageLabel: "Rows per page",
      jumpToPageDropdownLabel: "Jump to Page Dropdown",
      jumpToPageInputLabel: "Jump to Page Input",
      selectRow: "Row Selected",
      unselectRow: "Row Unselected",
      expandRow: "Row Expanded",
      collapseRow: "Row Collapsed",
      showFilterMenu: "Show Filter Menu",
      hideFilterMenu: "Hide Filter Menu",
      filterOperator: "Filter Operator",
      filterConstraint: "Filter Constraint",
      editRow: "Row Edit",
      saveEdit: "Save Edit",
      cancelEdit: "Cancel Edit",
      listView: "List View",
      gridView: "Grid View",
      slide: "Slide",
      slideNumber: "{slideNumber}",
      zoomImage: "Zoom Image",
      zoomIn: "Zoom In",
      zoomOut: "Zoom Out",
      rotateRight: "Rotate Right",
      rotateLeft: "Rotate Left",
      listLabel: "Option List"
    }
  },
  filterMatchModeOptions: {
    text: [je.STARTS_WITH, je.CONTAINS, je.NOT_CONTAINS, je.ENDS_WITH, je.EQUALS, je.NOT_EQUALS],
    numeric: [je.EQUALS, je.NOT_EQUALS, je.LESS_THAN, je.LESS_THAN_OR_EQUAL_TO, je.GREATER_THAN, je.GREATER_THAN_OR_EQUAL_TO],
    date: [je.DATE_IS, je.DATE_IS_NOT, je.DATE_BEFORE, je.DATE_AFTER]
  },
  zIndex: {
    modal: 1100,
    overlay: 1e3,
    menu: 1e3,
    tooltip: 1100
  },
  theme: void 0,
  unstyled: !1,
  pt: void 0,
  ptOptions: {
    mergeSections: !0,
    mergeProps: !1
  },
  csp: {
    nonce: void 0
  }
}, fm = Symbol();
function pm(e, t) {
  var n = {
    config: To(t)
  };
  return e.config.globalProperties.$primevue = n, e.provide(fm, n), mm(), hm(e, n), n;
}
var Rn = [];
function mm() {
  xe.clear(), Rn.forEach(function(e) {
    return e == null ? void 0 : e();
  }), Rn = [];
}
function hm(e, t) {
  var n = De(!1), r = function() {
    var u;
    if (((u = t.config) === null || u === void 0 ? void 0 : u.theme) !== "none" && !fe.isStyleNameLoaded("common")) {
      var c, d, f = ((c = ye.getCommonTheme) === null || c === void 0 ? void 0 : c.call(ye)) || {}, h = f.primitive, _ = f.semantic, E = f.global, w = f.style, P = {
        nonce: (d = t.config) === null || d === void 0 || (d = d.csp) === null || d === void 0 ? void 0 : d.nonce
      };
      ye.load(h == null ? void 0 : h.css, Zr({
        name: "primitive-variables"
      }, P)), ye.load(_ == null ? void 0 : _.css, Zr({
        name: "semantic-variables"
      }, P)), ye.load(E == null ? void 0 : E.css, Zr({
        name: "global-variables"
      }, P)), ye.loadStyle(Zr({
        name: "global-style"
      }, P), w), fe.setLoadedStyleName("common");
    }
  };
  xe.on("theme:change", function(l) {
    n.value || (e.config.globalProperties.$primevue.config.theme = l, n.value = !0);
  });
  var o = ft(t.config, function(l, u) {
    an.emit("config:change", {
      newValue: l,
      oldValue: u
    });
  }, {
    immediate: !0,
    deep: !0
  }), i = ft(function() {
    return t.config.ripple;
  }, function(l, u) {
    an.emit("config:ripple:change", {
      newValue: l,
      oldValue: u
    });
  }, {
    immediate: !0,
    deep: !0
  }), s = ft(function() {
    return t.config.theme;
  }, function(l, u) {
    n.value || fe.setTheme(l), t.config.unstyled || r(), n.value = !1, an.emit("config:theme:change", {
      newValue: l,
      oldValue: u
    });
  }, {
    immediate: !0,
    deep: !1
  }), a = ft(function() {
    return t.config.unstyled;
  }, function(l, u) {
    !l && t.config.theme && r(), an.emit("config:unstyled:change", {
      newValue: l,
      oldValue: u
    });
  }, {
    immediate: !0,
    deep: !0
  });
  Rn.push(o), Rn.push(i), Rn.push(s), Rn.push(a);
}
var gm = {
  install: function(t, n) {
    var r = yp(dm, n);
    pm(t, r);
  }
}, sn = {
  _loadedStyleNames: /* @__PURE__ */ new Set(),
  getLoadedStyleNames: function() {
    return this._loadedStyleNames;
  },
  isStyleNameLoaded: function(t) {
    return this._loadedStyleNames.has(t);
  },
  setLoadedStyleName: function(t) {
    this._loadedStyleNames.add(t);
  },
  deleteLoadedStyleName: function(t) {
    this._loadedStyleNames.delete(t);
  },
  clearLoadedStyleNames: function() {
    this._loadedStyleNames.clear();
  }
};
function bm() {
  var e = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "pc", t = xd();
  return "".concat(e).concat(t.replace("v-", "").replaceAll("-", "_"));
}
var ga = ye.extend({
  name: "common"
});
function Pr(e) {
  "@babel/helpers - typeof";
  return Pr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Pr(e);
}
function vm(e) {
  return qu(e) || ym(e) || Ju(e) || Xu();
}
function ym(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function qn(e, t) {
  return qu(e) || _m(e, t) || Ju(e, t) || Xu();
}
function Xu() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Ju(e, t) {
  if (e) {
    if (typeof e == "string") return ba(e, t);
    var n = {}.toString.call(e).slice(8, -1);
    return n === "Object" && e.constructor && (n = e.constructor.name), n === "Map" || n === "Set" ? Array.from(e) : n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? ba(e, t) : void 0;
  }
}
function ba(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var n = 0, r = Array(t); n < t; n++) r[n] = e[n];
  return r;
}
function _m(e, t) {
  var n = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (n != null) {
    var r, o, i, s, a = [], l = !0, u = !1;
    try {
      if (i = (n = n.call(e)).next, t === 0) {
        if (Object(n) !== n) return;
        l = !1;
      } else for (; !(l = (r = i.call(n)).done) && (a.push(r.value), a.length !== t); l = !0) ;
    } catch (c) {
      u = !0, o = c;
    } finally {
      try {
        if (!l && n.return != null && (s = n.return(), Object(s) !== s)) return;
      } finally {
        if (u) throw o;
      }
    }
    return a;
  }
}
function qu(e) {
  if (Array.isArray(e)) return e;
}
function va(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function oe(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? va(Object(n), !0).forEach(function(r) {
      nr(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : va(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function nr(e, t, n) {
  return (t = Sm(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function Sm(e) {
  var t = Em(e, "string");
  return Pr(t) == "symbol" ? t : t + "";
}
function Em(e, t) {
  if (Pr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Pr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var ko = {
  name: "BaseComponent",
  props: {
    pt: {
      type: Object,
      default: void 0
    },
    ptOptions: {
      type: Object,
      default: void 0
    },
    unstyled: {
      type: Boolean,
      default: void 0
    },
    dt: {
      type: Object,
      default: void 0
    }
  },
  inject: {
    $parentInstance: {
      default: void 0
    }
  },
  watch: {
    isUnstyled: {
      immediate: !0,
      handler: function(t) {
        xe.off("theme:change", this._loadCoreStyles), t || (this._loadCoreStyles(), this._themeChangeListener(this._loadCoreStyles));
      }
    },
    dt: {
      immediate: !0,
      handler: function(t, n) {
        var r = this;
        xe.off("theme:change", this._themeScopedListener), t ? (this._loadScopedThemeStyles(t), this._themeScopedListener = function() {
          return r._loadScopedThemeStyles(t);
        }, this._themeChangeListener(this._themeScopedListener)) : this._unloadScopedThemeStyles();
      }
    }
  },
  scopedStyleEl: void 0,
  rootEl: void 0,
  uid: void 0,
  $attrSelector: void 0,
  beforeCreate: function() {
    var t, n, r, o, i, s, a, l, u, c, d, f = (t = this.pt) === null || t === void 0 ? void 0 : t._usept, h = f ? (n = this.pt) === null || n === void 0 || (n = n.originalValue) === null || n === void 0 ? void 0 : n[this.$.type.name] : void 0, _ = f ? (r = this.pt) === null || r === void 0 || (r = r.value) === null || r === void 0 ? void 0 : r[this.$.type.name] : this.pt;
    (o = _ || h) === null || o === void 0 || (o = o.hooks) === null || o === void 0 || (i = o.onBeforeCreate) === null || i === void 0 || i.call(o);
    var E = (s = this.$primevueConfig) === null || s === void 0 || (s = s.pt) === null || s === void 0 ? void 0 : s._usept, w = E ? (a = this.$primevue) === null || a === void 0 || (a = a.config) === null || a === void 0 || (a = a.pt) === null || a === void 0 ? void 0 : a.originalValue : void 0, P = E ? (l = this.$primevue) === null || l === void 0 || (l = l.config) === null || l === void 0 || (l = l.pt) === null || l === void 0 ? void 0 : l.value : (u = this.$primevue) === null || u === void 0 || (u = u.config) === null || u === void 0 ? void 0 : u.pt;
    (c = P || w) === null || c === void 0 || (c = c[this.$.type.name]) === null || c === void 0 || (c = c.hooks) === null || c === void 0 || (d = c.onBeforeCreate) === null || d === void 0 || d.call(c), this.$attrSelector = bm(), this.uid = this.$attrs.id || this.$attrSelector.replace("pc", "pv_id_");
  },
  created: function() {
    this._hook("onCreated");
  },
  beforeMount: function() {
    var t;
    this.rootEl = $p($n(this.$el) ? this.$el : (t = this.$el) === null || t === void 0 ? void 0 : t.parentElement, "[".concat(this.$attrSelector, "]")), this.rootEl && (this.rootEl.$pc = oe({
      name: this.$.type.name,
      attrSelector: this.$attrSelector
    }, this.$params)), this._loadStyles(), this._hook("onBeforeMount");
  },
  mounted: function() {
    this._hook("onMounted");
  },
  beforeUpdate: function() {
    this._hook("onBeforeUpdate");
  },
  updated: function() {
    this._hook("onUpdated");
  },
  beforeUnmount: function() {
    this._hook("onBeforeUnmount");
  },
  unmounted: function() {
    this._removeThemeListeners(), this._unloadScopedThemeStyles(), this._hook("onUnmounted");
  },
  methods: {
    _hook: function(t) {
      if (!this.$options.hostName) {
        var n = this._usePT(this._getPT(this.pt, this.$.type.name), this._getOptionValue, "hooks.".concat(t)), r = this._useDefaultPT(this._getOptionValue, "hooks.".concat(t));
        n == null || n(), r == null || r();
      }
    },
    _mergeProps: function(t) {
      for (var n = arguments.length, r = new Array(n > 1 ? n - 1 : 0), o = 1; o < n; o++)
        r[o - 1] = arguments[o];
      return es(t) ? t.apply(void 0, r) : de.apply(void 0, r);
    },
    _load: function() {
      sn.isStyleNameLoaded("base") || (ye.loadCSS(this.$styleOptions), this._loadGlobalStyles(), sn.setLoadedStyleName("base")), this._loadThemeStyles();
    },
    _loadStyles: function() {
      this._load(), this._themeChangeListener(this._load);
    },
    _loadCoreStyles: function() {
      var t, n;
      !sn.isStyleNameLoaded((t = this.$style) === null || t === void 0 ? void 0 : t.name) && (n = this.$style) !== null && n !== void 0 && n.name && (ga.loadCSS(this.$styleOptions), this.$options.style && this.$style.loadCSS(this.$styleOptions), sn.setLoadedStyleName(this.$style.name));
    },
    _loadGlobalStyles: function() {
      var t = this._useGlobalPT(this._getOptionValue, "global.css", this.$params);
      _e(t) && ye.load(t, oe({
        name: "global"
      }, this.$styleOptions));
    },
    _loadThemeStyles: function() {
      var t, n;
      if (!(this.isUnstyled || this.$theme === "none")) {
        if (!fe.isStyleNameLoaded("common")) {
          var r, o, i = ((r = this.$style) === null || r === void 0 || (o = r.getCommonTheme) === null || o === void 0 ? void 0 : o.call(r)) || {}, s = i.primitive, a = i.semantic, l = i.global, u = i.style;
          ye.load(s == null ? void 0 : s.css, oe({
            name: "primitive-variables"
          }, this.$styleOptions)), ye.load(a == null ? void 0 : a.css, oe({
            name: "semantic-variables"
          }, this.$styleOptions)), ye.load(l == null ? void 0 : l.css, oe({
            name: "global-variables"
          }, this.$styleOptions)), ye.loadStyle(oe({
            name: "global-style"
          }, this.$styleOptions), u), fe.setLoadedStyleName("common");
        }
        if (!fe.isStyleNameLoaded((t = this.$style) === null || t === void 0 ? void 0 : t.name) && (n = this.$style) !== null && n !== void 0 && n.name) {
          var c, d, f, h, _ = ((c = this.$style) === null || c === void 0 || (d = c.getComponentTheme) === null || d === void 0 ? void 0 : d.call(c)) || {}, E = _.css, w = _.style;
          (f = this.$style) === null || f === void 0 || f.load(E, oe({
            name: "".concat(this.$style.name, "-variables")
          }, this.$styleOptions)), (h = this.$style) === null || h === void 0 || h.loadStyle(oe({
            name: "".concat(this.$style.name, "-style")
          }, this.$styleOptions), w), fe.setLoadedStyleName(this.$style.name);
        }
        if (!fe.isStyleNameLoaded("layer-order")) {
          var P, M, S = (P = this.$style) === null || P === void 0 || (M = P.getLayerOrderThemeCSS) === null || M === void 0 ? void 0 : M.call(P);
          ye.load(S, oe({
            name: "layer-order",
            first: !0
          }, this.$styleOptions)), fe.setLoadedStyleName("layer-order");
        }
      }
    },
    _loadScopedThemeStyles: function(t) {
      var n, r, o, i = ((n = this.$style) === null || n === void 0 || (r = n.getPresetTheme) === null || r === void 0 ? void 0 : r.call(n, t, "[".concat(this.$attrSelector, "]"))) || {}, s = i.css, a = (o = this.$style) === null || o === void 0 ? void 0 : o.load(s, oe({
        name: "".concat(this.$attrSelector, "-").concat(this.$style.name)
      }, this.$styleOptions));
      this.scopedStyleEl = a.el;
    },
    _unloadScopedThemeStyles: function() {
      var t;
      (t = this.scopedStyleEl) === null || t === void 0 || (t = t.value) === null || t === void 0 || t.remove();
    },
    _themeChangeListener: function() {
      var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : function() {
      };
      sn.clearLoadedStyleNames(), xe.on("theme:change", t);
    },
    _removeThemeListeners: function() {
      xe.off("theme:change", this._loadCoreStyles), xe.off("theme:change", this._load), xe.off("theme:change", this._themeScopedListener);
    },
    _getHostInstance: function(t) {
      return t ? this.$options.hostName ? t.$.type.name === this.$options.hostName ? t : this._getHostInstance(t.$parentInstance) : t.$parentInstance : void 0;
    },
    _getPropValue: function(t) {
      var n;
      return this[t] || ((n = this._getHostInstance(this)) === null || n === void 0 ? void 0 : n[t]);
    },
    _getOptionValue: function(t) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      return ts(t, n, r);
    },
    _getPTValue: function() {
      var t, n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", o = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {}, i = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : !0, s = /./g.test(r) && !!o[r.split(".")[0]], a = this._getPropValue("ptOptions") || ((t = this.$primevueConfig) === null || t === void 0 ? void 0 : t.ptOptions) || {}, l = a.mergeSections, u = l === void 0 ? !0 : l, c = a.mergeProps, d = c === void 0 ? !1 : c, f = i ? s ? this._useGlobalPT(this._getPTClassValue, r, o) : this._useDefaultPT(this._getPTClassValue, r, o) : void 0, h = s ? void 0 : this._getPTSelf(n, this._getPTClassValue, r, oe(oe({}, o), {}, {
        global: f || {}
      })), _ = this._getPTDatasets(r);
      return u || !u && h ? d ? this._mergeProps(d, f, h, _) : oe(oe(oe({}, f), h), _) : oe(oe({}, h), _);
    },
    _getPTSelf: function() {
      for (var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length, r = new Array(n > 1 ? n - 1 : 0), o = 1; o < n; o++)
        r[o - 1] = arguments[o];
      return de(
        this._usePT.apply(this, [this._getPT(t, this.$name)].concat(r)),
        // Exp; <component :pt="{}"
        this._usePT.apply(this, [this.$_attrsPT].concat(r))
        // Exp; <component :pt:[passthrough_key]:[attribute]="{value}" or <component :pt:[passthrough_key]="() =>{value}"
      );
    },
    _getPTDatasets: function() {
      var t, n, r = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", o = "data-pc-", i = r === "root" && _e((t = this.pt) === null || t === void 0 ? void 0 : t["data-pc-section"]);
      return r !== "transition" && oe(oe({}, r === "root" && oe(oe(nr({}, "".concat(o, "name"), At(i ? (n = this.pt) === null || n === void 0 ? void 0 : n["data-pc-section"] : this.$.type.name)), i && nr({}, "".concat(o, "extend"), At(this.$.type.name))), {}, nr({}, "".concat(this.$attrSelector), ""))), {}, nr({}, "".concat(o, "section"), At(r)));
    },
    _getPTClassValue: function() {
      var t = this._getOptionValue.apply(this, arguments);
      return ot(t) || Iu(t) ? {
        class: t
      } : t;
    },
    _getPT: function(t) {
      var n = this, r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", o = arguments.length > 2 ? arguments[2] : void 0, i = function(a) {
        var l, u = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !1, c = o ? o(a) : a, d = At(r), f = At(n.$name);
        return (l = u ? d !== f ? c == null ? void 0 : c[d] : void 0 : c == null ? void 0 : c[d]) !== null && l !== void 0 ? l : c;
      };
      return t != null && t.hasOwnProperty("_usept") ? {
        _usept: t._usept,
        originalValue: i(t.originalValue),
        value: i(t.value)
      } : i(t, !0);
    },
    _usePT: function(t, n, r, o) {
      var i = function(E) {
        return n(E, r, o);
      };
      if (t != null && t.hasOwnProperty("_usept")) {
        var s, a = t._usept || ((s = this.$primevueConfig) === null || s === void 0 ? void 0 : s.ptOptions) || {}, l = a.mergeSections, u = l === void 0 ? !0 : l, c = a.mergeProps, d = c === void 0 ? !1 : c, f = i(t.originalValue), h = i(t.value);
        return f === void 0 && h === void 0 ? void 0 : ot(h) ? h : ot(f) ? f : u || !u && h ? d ? this._mergeProps(d, f, h) : oe(oe({}, f), h) : h;
      }
      return i(t);
    },
    _useGlobalPT: function(t, n, r) {
      return this._usePT(this.globalPT, t, n, r);
    },
    _useDefaultPT: function(t, n, r) {
      return this._usePT(this.defaultPT, t, n, r);
    },
    ptm: function() {
      var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
      return this._getPTValue(this.pt, t, oe(oe({}, this.$params), n));
    },
    ptmi: function() {
      var t, n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, o = de(this.$_attrsWithoutPT, this.ptm(n, r));
      return o != null && o.hasOwnProperty("id") && ((t = o.id) !== null && t !== void 0 || (o.id = this.$id)), o;
    },
    ptmo: function() {
      var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      return this._getPTValue(t, n, oe({
        instance: this
      }, r), !1);
    },
    cx: function() {
      var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
      return this.isUnstyled ? void 0 : this._getOptionValue(this.$style.classes, t, oe(oe({}, this.$params), n));
    },
    sx: function() {
      var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !0, r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      if (n) {
        var o = this._getOptionValue(this.$style.inlineStyles, t, oe(oe({}, this.$params), r)), i = this._getOptionValue(ga.inlineStyles, t, oe(oe({}, this.$params), r));
        return [i, o];
      }
    }
  },
  computed: {
    globalPT: function() {
      var t, n = this;
      return this._getPT((t = this.$primevueConfig) === null || t === void 0 ? void 0 : t.pt, void 0, function(r) {
        return lt(r, {
          instance: n
        });
      });
    },
    defaultPT: function() {
      var t, n = this;
      return this._getPT((t = this.$primevueConfig) === null || t === void 0 ? void 0 : t.pt, void 0, function(r) {
        return n._getOptionValue(r, n.$name, oe({}, n.$params)) || lt(r, oe({}, n.$params));
      });
    },
    isUnstyled: function() {
      var t;
      return this.unstyled !== void 0 ? this.unstyled : (t = this.$primevueConfig) === null || t === void 0 ? void 0 : t.unstyled;
    },
    $id: function() {
      return this.$attrs.id || this.uid;
    },
    $inProps: function() {
      var t, n = Object.keys(((t = this.$.vnode) === null || t === void 0 ? void 0 : t.props) || {});
      return Object.fromEntries(Object.entries(this.$props).filter(function(r) {
        var o = qn(r, 1), i = o[0];
        return n == null ? void 0 : n.includes(i);
      }));
    },
    $theme: function() {
      var t;
      return (t = this.$primevueConfig) === null || t === void 0 ? void 0 : t.theme;
    },
    $style: function() {
      return oe(oe({
        classes: void 0,
        inlineStyles: void 0,
        load: function() {
        },
        loadCSS: function() {
        },
        loadStyle: function() {
        }
      }, (this._getHostInstance(this) || {}).$style), this.$options.style);
    },
    $styleOptions: function() {
      var t;
      return {
        nonce: (t = this.$primevueConfig) === null || t === void 0 || (t = t.csp) === null || t === void 0 ? void 0 : t.nonce
      };
    },
    $primevueConfig: function() {
      var t;
      return (t = this.$primevue) === null || t === void 0 ? void 0 : t.config;
    },
    $name: function() {
      return this.$options.hostName || this.$.type.name;
    },
    $params: function() {
      var t = this._getHostInstance(this) || this.$parent;
      return {
        instance: this,
        props: this.$props,
        state: this.$data,
        attrs: this.$attrs,
        parent: {
          instance: t,
          props: t == null ? void 0 : t.$props,
          state: t == null ? void 0 : t.$data,
          attrs: t == null ? void 0 : t.$attrs
        }
      };
    },
    $_attrsPT: function() {
      return Object.entries(this.$attrs || {}).filter(function(t) {
        var n = qn(t, 1), r = n[0];
        return r == null ? void 0 : r.startsWith("pt:");
      }).reduce(function(t, n) {
        var r = qn(n, 2), o = r[0], i = r[1], s = o.split(":"), a = vm(s), l = a.slice(1);
        return l == null || l.reduce(function(u, c, d, f) {
          return !u[c] && (u[c] = d === f.length - 1 ? i : {}), u[c];
        }, t), t;
      }, {});
    },
    $_attrsWithoutPT: function() {
      return Object.entries(this.$attrs || {}).filter(function(t) {
        var n = qn(t, 1), r = n[0];
        return !(r != null && r.startsWith("pt:"));
      }).reduce(function(t, n) {
        var r = qn(n, 2), o = r[0], i = r[1];
        return t[o] = i, t;
      }, {});
    }
  }
}, Tm = `
.p-icon {
    display: inline-block;
    vertical-align: baseline;
}

.p-icon-spin {
    -webkit-animation: p-icon-spin 2s infinite linear;
    animation: p-icon-spin 2s infinite linear;
}

@-webkit-keyframes p-icon-spin {
    0% {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
    100% {
        -webkit-transform: rotate(359deg);
        transform: rotate(359deg);
    }
}

@keyframes p-icon-spin {
    0% {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
    100% {
        -webkit-transform: rotate(359deg);
        transform: rotate(359deg);
    }
}
`, Cm = ye.extend({
  name: "baseicon",
  css: Tm
});
function wr(e) {
  "@babel/helpers - typeof";
  return wr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, wr(e);
}
function ya(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function _a(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ya(Object(n), !0).forEach(function(r) {
      Om(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : ya(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function Om(e, t, n) {
  return (t = Lm(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function Lm(e) {
  var t = Pm(e, "string");
  return wr(t) == "symbol" ? t : t + "";
}
function Pm(e, t) {
  if (wr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (wr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var $o = {
  name: "BaseIcon",
  extends: ko,
  props: {
    label: {
      type: String,
      default: void 0
    },
    spin: {
      type: Boolean,
      default: !1
    }
  },
  style: Cm,
  provide: function() {
    return {
      $pcIcon: this,
      $parentInstance: this
    };
  },
  methods: {
    pti: function() {
      var t = kn(this.label);
      return _a(_a({}, !this.isUnstyled && {
        class: ["p-icon", {
          "p-icon-spin": this.spin
        }]
      }), {}, {
        role: t ? void 0 : "img",
        "aria-label": t ? void 0 : this.label,
        "aria-hidden": t
      });
    }
  }
}, Zu = {
  name: "TimesIcon",
  extends: $o
};
function wm(e, t, n, r, o, i) {
  return be(), Ze("svg", de({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, e.pti()), t[0] || (t[0] = [Ft("path", {
    d: "M8.01186 7.00933L12.27 2.75116C12.341 2.68501 12.398 2.60524 12.4375 2.51661C12.4769 2.42798 12.4982 2.3323 12.4999 2.23529C12.5016 2.13827 12.4838 2.0419 12.4474 1.95194C12.4111 1.86197 12.357 1.78024 12.2884 1.71163C12.2198 1.64302 12.138 1.58893 12.0481 1.55259C11.9581 1.51625 11.8617 1.4984 11.7647 1.50011C11.6677 1.50182 11.572 1.52306 11.4834 1.56255C11.3948 1.60204 11.315 1.65898 11.2488 1.72997L6.99067 5.98814L2.7325 1.72997C2.59553 1.60234 2.41437 1.53286 2.22718 1.53616C2.03999 1.53946 1.8614 1.61529 1.72901 1.74767C1.59663 1.88006 1.5208 2.05865 1.5175 2.24584C1.5142 2.43303 1.58368 2.61419 1.71131 2.75116L5.96948 7.00933L1.71131 11.2675C1.576 11.403 1.5 11.5866 1.5 11.7781C1.5 11.9696 1.576 12.1532 1.71131 12.2887C1.84679 12.424 2.03043 12.5 2.2219 12.5C2.41338 12.5 2.59702 12.424 2.7325 12.2887L6.99067 8.03052L11.2488 12.2887C11.3843 12.424 11.568 12.5 11.7594 12.5C11.9509 12.5 12.1346 12.424 12.27 12.2887C12.4053 12.1532 12.4813 11.9696 12.4813 11.7781C12.4813 11.5866 12.4053 11.403 12.27 11.2675L8.01186 7.00933Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
Zu.render = wm;
var Qu = {
  name: "WindowMaximizeIcon",
  extends: $o
};
function km(e, t, n, r, o, i) {
  return be(), Ze("svg", de({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, e.pti()), t[0] || (t[0] = [Ft("path", {
    "fill-rule": "evenodd",
    "clip-rule": "evenodd",
    d: "M7 14H11.8C12.3835 14 12.9431 13.7682 13.3556 13.3556C13.7682 12.9431 14 12.3835 14 11.8V2.2C14 1.61652 13.7682 1.05694 13.3556 0.644365C12.9431 0.231785 12.3835 0 11.8 0H2.2C1.61652 0 1.05694 0.231785 0.644365 0.644365C0.231785 1.05694 0 1.61652 0 2.2V7C0 7.15913 0.063214 7.31174 0.175736 7.42426C0.288258 7.53679 0.44087 7.6 0.6 7.6C0.75913 7.6 0.911742 7.53679 1.02426 7.42426C1.13679 7.31174 1.2 7.15913 1.2 7V2.2C1.2 1.93478 1.30536 1.68043 1.49289 1.49289C1.68043 1.30536 1.93478 1.2 2.2 1.2H11.8C12.0652 1.2 12.3196 1.30536 12.5071 1.49289C12.6946 1.68043 12.8 1.93478 12.8 2.2V11.8C12.8 12.0652 12.6946 12.3196 12.5071 12.5071C12.3196 12.6946 12.0652 12.8 11.8 12.8H7C6.84087 12.8 6.68826 12.8632 6.57574 12.9757C6.46321 13.0883 6.4 13.2409 6.4 13.4C6.4 13.5591 6.46321 13.7117 6.57574 13.8243C6.68826 13.9368 6.84087 14 7 14ZM9.77805 7.42192C9.89013 7.534 10.0415 7.59788 10.2 7.59995C10.3585 7.59788 10.5099 7.534 10.622 7.42192C10.7341 7.30985 10.798 7.15844 10.8 6.99995V3.94242C10.8066 3.90505 10.8096 3.86689 10.8089 3.82843C10.8079 3.77159 10.7988 3.7157 10.7824 3.6623C10.756 3.55552 10.701 3.45698 10.622 3.37798C10.5099 3.2659 10.3585 3.20202 10.2 3.19995H7.00002C6.84089 3.19995 6.68828 3.26317 6.57576 3.37569C6.46324 3.48821 6.40002 3.64082 6.40002 3.79995C6.40002 3.95908 6.46324 4.11169 6.57576 4.22422C6.68828 4.33674 6.84089 4.39995 7.00002 4.39995H8.80006L6.19997 7.00005C6.10158 7.11005 6.04718 7.25246 6.04718 7.40005C6.04718 7.54763 6.10158 7.69004 6.19997 7.80005C6.30202 7.91645 6.44561 7.98824 6.59997 8.00005C6.75432 7.98824 6.89791 7.91645 6.99997 7.80005L9.60002 5.26841V6.99995C9.6021 7.15844 9.66598 7.30985 9.77805 7.42192ZM1.4 14H3.8C4.17066 13.9979 4.52553 13.8498 4.78763 13.5877C5.04973 13.3256 5.1979 12.9707 5.2 12.6V10.2C5.1979 9.82939 5.04973 9.47452 4.78763 9.21242C4.52553 8.95032 4.17066 8.80215 3.8 8.80005H1.4C1.02934 8.80215 0.674468 8.95032 0.412371 9.21242C0.150274 9.47452 0.00210008 9.82939 0 10.2V12.6C0.00210008 12.9707 0.150274 13.3256 0.412371 13.5877C0.674468 13.8498 1.02934 13.9979 1.4 14ZM1.25858 10.0586C1.29609 10.0211 1.34696 10 1.4 10H3.8C3.85304 10 3.90391 10.0211 3.94142 10.0586C3.97893 10.0961 4 10.147 4 10.2V12.6C4 12.6531 3.97893 12.704 3.94142 12.7415C3.90391 12.779 3.85304 12.8 3.8 12.8H1.4C1.34696 12.8 1.29609 12.779 1.25858 12.7415C1.22107 12.704 1.2 12.6531 1.2 12.6V10.2C1.2 10.147 1.22107 10.0961 1.25858 10.0586Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
Qu.render = km;
var ec = {
  name: "WindowMinimizeIcon",
  extends: $o
};
function $m(e, t, n, r, o, i) {
  return be(), Ze("svg", de({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, e.pti()), t[0] || (t[0] = [Ft("path", {
    "fill-rule": "evenodd",
    "clip-rule": "evenodd",
    d: "M11.8 0H2.2C1.61652 0 1.05694 0.231785 0.644365 0.644365C0.231785 1.05694 0 1.61652 0 2.2V7C0 7.15913 0.063214 7.31174 0.175736 7.42426C0.288258 7.53679 0.44087 7.6 0.6 7.6C0.75913 7.6 0.911742 7.53679 1.02426 7.42426C1.13679 7.31174 1.2 7.15913 1.2 7V2.2C1.2 1.93478 1.30536 1.68043 1.49289 1.49289C1.68043 1.30536 1.93478 1.2 2.2 1.2H11.8C12.0652 1.2 12.3196 1.30536 12.5071 1.49289C12.6946 1.68043 12.8 1.93478 12.8 2.2V11.8C12.8 12.0652 12.6946 12.3196 12.5071 12.5071C12.3196 12.6946 12.0652 12.8 11.8 12.8H7C6.84087 12.8 6.68826 12.8632 6.57574 12.9757C6.46321 13.0883 6.4 13.2409 6.4 13.4C6.4 13.5591 6.46321 13.7117 6.57574 13.8243C6.68826 13.9368 6.84087 14 7 14H11.8C12.3835 14 12.9431 13.7682 13.3556 13.3556C13.7682 12.9431 14 12.3835 14 11.8V2.2C14 1.61652 13.7682 1.05694 13.3556 0.644365C12.9431 0.231785 12.3835 0 11.8 0ZM6.368 7.952C6.44137 7.98326 6.52025 7.99958 6.6 8H9.8C9.95913 8 10.1117 7.93678 10.2243 7.82426C10.3368 7.71174 10.4 7.55913 10.4 7.4C10.4 7.24087 10.3368 7.08826 10.2243 6.97574C10.1117 6.86321 9.95913 6.8 9.8 6.8H8.048L10.624 4.224C10.73 4.11026 10.7877 3.95982 10.7849 3.80438C10.7822 3.64894 10.7192 3.50063 10.6093 3.3907C10.4994 3.28077 10.3511 3.2178 10.1956 3.21506C10.0402 3.21232 9.88974 3.27002 9.776 3.376L7.2 5.952V4.2C7.2 4.04087 7.13679 3.88826 7.02426 3.77574C6.91174 3.66321 6.75913 3.6 6.6 3.6C6.44087 3.6 6.28826 3.66321 6.17574 3.77574C6.06321 3.88826 6 4.04087 6 4.2V7.4C6.00042 7.47975 6.01674 7.55862 6.048 7.632C6.07656 7.70442 6.11971 7.7702 6.17475 7.82524C6.2298 7.88029 6.29558 7.92344 6.368 7.952ZM1.4 8.80005H3.8C4.17066 8.80215 4.52553 8.95032 4.78763 9.21242C5.04973 9.47452 5.1979 9.82939 5.2 10.2V12.6C5.1979 12.9707 5.04973 13.3256 4.78763 13.5877C4.52553 13.8498 4.17066 13.9979 3.8 14H1.4C1.02934 13.9979 0.674468 13.8498 0.412371 13.5877C0.150274 13.3256 0.00210008 12.9707 0 12.6V10.2C0.00210008 9.82939 0.150274 9.47452 0.412371 9.21242C0.674468 8.95032 1.02934 8.80215 1.4 8.80005ZM3.94142 12.7415C3.97893 12.704 4 12.6531 4 12.6V10.2C4 10.147 3.97893 10.0961 3.94142 10.0586C3.90391 10.0211 3.85304 10 3.8 10H1.4C1.34696 10 1.29609 10.0211 1.25858 10.0586C1.22107 10.0961 1.2 10.147 1.2 10.2V12.6C1.2 12.6531 1.22107 12.704 1.25858 12.7415C1.29609 12.779 1.34696 12.8 1.4 12.8H3.8C3.85304 12.8 3.90391 12.779 3.94142 12.7415Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
ec.render = $m;
var tc = {
  name: "SpinnerIcon",
  extends: $o
};
function Nm(e, t, n, r, o, i) {
  return be(), Ze("svg", de({
    width: "14",
    height: "14",
    viewBox: "0 0 14 14",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg"
  }, e.pti()), t[0] || (t[0] = [Ft("path", {
    d: "M6.99701 14C5.85441 13.999 4.72939 13.7186 3.72012 13.1832C2.71084 12.6478 1.84795 11.8737 1.20673 10.9284C0.565504 9.98305 0.165424 8.89526 0.041387 7.75989C-0.0826496 6.62453 0.073125 5.47607 0.495122 4.4147C0.917119 3.35333 1.59252 2.4113 2.46241 1.67077C3.33229 0.930247 4.37024 0.413729 5.4857 0.166275C6.60117 -0.0811796 7.76026 -0.0520535 8.86188 0.251112C9.9635 0.554278 10.9742 1.12227 11.8057 1.90555C11.915 2.01493 11.9764 2.16319 11.9764 2.31778C11.9764 2.47236 11.915 2.62062 11.8057 2.73C11.7521 2.78503 11.688 2.82877 11.6171 2.85864C11.5463 2.8885 11.4702 2.90389 11.3933 2.90389C11.3165 2.90389 11.2404 2.8885 11.1695 2.85864C11.0987 2.82877 11.0346 2.78503 10.9809 2.73C9.9998 1.81273 8.73246 1.26138 7.39226 1.16876C6.05206 1.07615 4.72086 1.44794 3.62279 2.22152C2.52471 2.99511 1.72683 4.12325 1.36345 5.41602C1.00008 6.70879 1.09342 8.08723 1.62775 9.31926C2.16209 10.5513 3.10478 11.5617 4.29713 12.1803C5.48947 12.7989 6.85865 12.988 8.17414 12.7157C9.48963 12.4435 10.6711 11.7264 11.5196 10.6854C12.3681 9.64432 12.8319 8.34282 12.8328 7C12.8328 6.84529 12.8943 6.69692 13.0038 6.58752C13.1132 6.47812 13.2616 6.41667 13.4164 6.41667C13.5712 6.41667 13.7196 6.47812 13.8291 6.58752C13.9385 6.69692 14 6.84529 14 7C14 8.85651 13.2622 10.637 11.9489 11.9497C10.6356 13.2625 8.85432 14 6.99701 14Z",
    fill: "currentColor"
  }, null, -1)]), 16);
}
tc.render = Nm;
var Im = `
    .p-badge {
        display: inline-flex;
        border-radius: dt('badge.border.radius');
        align-items: center;
        justify-content: center;
        padding: dt('badge.padding');
        background: dt('badge.primary.background');
        color: dt('badge.primary.color');
        font-size: dt('badge.font.size');
        font-weight: dt('badge.font.weight');
        min-width: dt('badge.min.width');
        height: dt('badge.height');
    }

    .p-badge-dot {
        width: dt('badge.dot.size');
        min-width: dt('badge.dot.size');
        height: dt('badge.dot.size');
        border-radius: 50%;
        padding: 0;
    }

    .p-badge-circle {
        padding: 0;
        border-radius: 50%;
    }

    .p-badge-secondary {
        background: dt('badge.secondary.background');
        color: dt('badge.secondary.color');
    }

    .p-badge-success {
        background: dt('badge.success.background');
        color: dt('badge.success.color');
    }

    .p-badge-info {
        background: dt('badge.info.background');
        color: dt('badge.info.color');
    }

    .p-badge-warn {
        background: dt('badge.warn.background');
        color: dt('badge.warn.color');
    }

    .p-badge-danger {
        background: dt('badge.danger.background');
        color: dt('badge.danger.color');
    }

    .p-badge-contrast {
        background: dt('badge.contrast.background');
        color: dt('badge.contrast.color');
    }

    .p-badge-sm {
        font-size: dt('badge.sm.font.size');
        min-width: dt('badge.sm.min.width');
        height: dt('badge.sm.height');
    }

    .p-badge-lg {
        font-size: dt('badge.lg.font.size');
        min-width: dt('badge.lg.min.width');
        height: dt('badge.lg.height');
    }

    .p-badge-xl {
        font-size: dt('badge.xl.font.size');
        min-width: dt('badge.xl.min.width');
        height: dt('badge.xl.height');
    }
`, Am = {
  root: function(t) {
    var n = t.props, r = t.instance;
    return ["p-badge p-component", {
      "p-badge-circle": _e(n.value) && String(n.value).length === 1,
      "p-badge-dot": kn(n.value) && !r.$slots.default,
      "p-badge-sm": n.size === "small",
      "p-badge-lg": n.size === "large",
      "p-badge-xl": n.size === "xlarge",
      "p-badge-info": n.severity === "info",
      "p-badge-success": n.severity === "success",
      "p-badge-warn": n.severity === "warn",
      "p-badge-danger": n.severity === "danger",
      "p-badge-secondary": n.severity === "secondary",
      "p-badge-contrast": n.severity === "contrast"
    }];
  }
}, xm = ye.extend({
  name: "badge",
  style: Im,
  classes: Am
}), Dm = {
  name: "BaseBadge",
  extends: ko,
  props: {
    value: {
      type: [String, Number],
      default: null
    },
    severity: {
      type: String,
      default: null
    },
    size: {
      type: String,
      default: null
    }
  },
  style: xm,
  provide: function() {
    return {
      $pcBadge: this,
      $parentInstance: this
    };
  }
};
function kr(e) {
  "@babel/helpers - typeof";
  return kr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, kr(e);
}
function Sa(e, t, n) {
  return (t = Rm(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function Rm(e) {
  var t = Mm(e, "string");
  return kr(t) == "symbol" ? t : t + "";
}
function Mm(e, t) {
  if (kr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (kr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var nc = {
  name: "Badge",
  extends: Dm,
  inheritAttrs: !1,
  computed: {
    dataP: function() {
      return Wn(Sa(Sa({
        circle: this.value != null && String(this.value).length === 1,
        empty: this.value == null && !this.$slots.default
      }, this.severity, this.severity), this.size, this.size));
    }
  }
}, Fm = ["data-p"];
function jm(e, t, n, r, o, i) {
  return be(), Ze("span", de({
    class: e.cx("root"),
    "data-p": i.dataP
  }, e.ptmi("root")), [Xe(e.$slots, "default", {}, function() {
    return [qi(mr(e.value), 1)];
  })], 16, Fm);
}
nc.render = jm;
function $r(e) {
  "@babel/helpers - typeof";
  return $r = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, $r(e);
}
function Ea(e, t) {
  return Wm(e) || Hm(e, t) || Vm(e, t) || Um();
}
function Um() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Vm(e, t) {
  if (e) {
    if (typeof e == "string") return Ta(e, t);
    var n = {}.toString.call(e).slice(8, -1);
    return n === "Object" && e.constructor && (n = e.constructor.name), n === "Map" || n === "Set" ? Array.from(e) : n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? Ta(e, t) : void 0;
  }
}
function Ta(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var n = 0, r = Array(t); n < t; n++) r[n] = e[n];
  return r;
}
function Hm(e, t) {
  var n = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (n != null) {
    var r, o, i, s, a = [], l = !0, u = !1;
    try {
      if (i = (n = n.call(e)).next, t !== 0) for (; !(l = (r = i.call(n)).done) && (a.push(r.value), a.length !== t); l = !0) ;
    } catch (c) {
      u = !0, o = c;
    } finally {
      try {
        if (!l && n.return != null && (s = n.return(), Object(s) !== s)) return;
      } finally {
        if (u) throw o;
      }
    }
    return a;
  }
}
function Wm(e) {
  if (Array.isArray(e)) return e;
}
function Ca(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function ie(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ca(Object(n), !0).forEach(function(r) {
      vi(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : Ca(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function vi(e, t, n) {
  return (t = Bm(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function Bm(e) {
  var t = Km(e, "string");
  return $r(t) == "symbol" ? t : t + "";
}
function Km(e, t) {
  if ($r(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if ($r(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Z = {
  _getMeta: function() {
    return [Rt(arguments.length <= 0 ? void 0 : arguments[0]) || arguments.length <= 0 ? void 0 : arguments[0], lt(Rt(arguments.length <= 0 ? void 0 : arguments[0]) ? arguments.length <= 0 ? void 0 : arguments[0] : arguments.length <= 1 ? void 0 : arguments[1])];
  },
  _getConfig: function(t, n) {
    var r, o, i;
    return (r = (t == null || (o = t.instance) === null || o === void 0 ? void 0 : o.$primevue) || (n == null || (i = n.ctx) === null || i === void 0 || (i = i.appContext) === null || i === void 0 || (i = i.config) === null || i === void 0 || (i = i.globalProperties) === null || i === void 0 ? void 0 : i.$primevue)) === null || r === void 0 ? void 0 : r.config;
  },
  _getOptionValue: ts,
  _getPTValue: function() {
    var t, n, r = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, i = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : "", s = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {}, a = arguments.length > 4 && arguments[4] !== void 0 ? arguments[4] : !0, l = function() {
      var M = Z._getOptionValue.apply(Z, arguments);
      return ot(M) || Iu(M) ? {
        class: M
      } : M;
    }, u = ((t = r.binding) === null || t === void 0 || (t = t.value) === null || t === void 0 ? void 0 : t.ptOptions) || ((n = r.$primevueConfig) === null || n === void 0 ? void 0 : n.ptOptions) || {}, c = u.mergeSections, d = c === void 0 ? !0 : c, f = u.mergeProps, h = f === void 0 ? !1 : f, _ = a ? Z._useDefaultPT(r, r.defaultPT(), l, i, s) : void 0, E = Z._usePT(r, Z._getPT(o, r.$name), l, i, ie(ie({}, s), {}, {
      global: _ || {}
    })), w = Z._getPTDatasets(r, i);
    return d || !d && E ? h ? Z._mergeProps(r, h, _, E, w) : ie(ie(ie({}, _), E), w) : ie(ie({}, E), w);
  },
  _getPTDatasets: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", r = "data-pc-";
    return ie(ie({}, n === "root" && vi({}, "".concat(r, "name"), At(t.$name))), {}, vi({}, "".concat(r, "section"), At(n)));
  },
  _getPT: function(t) {
    var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", r = arguments.length > 2 ? arguments[2] : void 0, o = function(s) {
      var a, l = r ? r(s) : s, u = At(n);
      return (a = l == null ? void 0 : l[u]) !== null && a !== void 0 ? a : l;
    };
    return t && Object.hasOwn(t, "_usept") ? {
      _usept: t._usept,
      originalValue: o(t.originalValue),
      value: o(t.value)
    } : o(t);
  },
  _usePT: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length > 1 ? arguments[1] : void 0, r = arguments.length > 2 ? arguments[2] : void 0, o = arguments.length > 3 ? arguments[3] : void 0, i = arguments.length > 4 ? arguments[4] : void 0, s = function(w) {
      return r(w, o, i);
    };
    if (n && Object.hasOwn(n, "_usept")) {
      var a, l = n._usept || ((a = t.$primevueConfig) === null || a === void 0 ? void 0 : a.ptOptions) || {}, u = l.mergeSections, c = u === void 0 ? !0 : u, d = l.mergeProps, f = d === void 0 ? !1 : d, h = s(n.originalValue), _ = s(n.value);
      return h === void 0 && _ === void 0 ? void 0 : ot(_) ? _ : ot(h) ? h : c || !c && _ ? f ? Z._mergeProps(t, f, h, _) : ie(ie({}, h), _) : _;
    }
    return s(n);
  },
  _useDefaultPT: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, r = arguments.length > 2 ? arguments[2] : void 0, o = arguments.length > 3 ? arguments[3] : void 0, i = arguments.length > 4 ? arguments[4] : void 0;
    return Z._usePT(t, n, r, o, i);
  },
  _loadStyles: function() {
    var t, n = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, r = arguments.length > 1 ? arguments[1] : void 0, o = arguments.length > 2 ? arguments[2] : void 0, i = Z._getConfig(r, o), s = {
      nonce: i == null || (t = i.csp) === null || t === void 0 ? void 0 : t.nonce
    };
    Z._loadCoreStyles(n, s), Z._loadThemeStyles(n, s), Z._loadScopedThemeStyles(n, s), Z._removeThemeListeners(n), n.$loadStyles = function() {
      return Z._loadThemeStyles(n, s);
    }, Z._themeChangeListener(n.$loadStyles);
  },
  _loadCoreStyles: function() {
    var t, n, r = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, o = arguments.length > 1 ? arguments[1] : void 0;
    if (!sn.isStyleNameLoaded((t = r.$style) === null || t === void 0 ? void 0 : t.name) && (n = r.$style) !== null && n !== void 0 && n.name) {
      var i;
      ye.loadCSS(o), (i = r.$style) === null || i === void 0 || i.loadCSS(o), sn.setLoadedStyleName(r.$style.name);
    }
  },
  _loadThemeStyles: function() {
    var t, n, r, o = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, i = arguments.length > 1 ? arguments[1] : void 0;
    if (!(o != null && o.isUnstyled() || (o == null || (t = o.theme) === null || t === void 0 ? void 0 : t.call(o)) === "none")) {
      if (!fe.isStyleNameLoaded("common")) {
        var s, a, l = ((s = o.$style) === null || s === void 0 || (a = s.getCommonTheme) === null || a === void 0 ? void 0 : a.call(s)) || {}, u = l.primitive, c = l.semantic, d = l.global, f = l.style;
        ye.load(u == null ? void 0 : u.css, ie({
          name: "primitive-variables"
        }, i)), ye.load(c == null ? void 0 : c.css, ie({
          name: "semantic-variables"
        }, i)), ye.load(d == null ? void 0 : d.css, ie({
          name: "global-variables"
        }, i)), ye.loadStyle(ie({
          name: "global-style"
        }, i), f), fe.setLoadedStyleName("common");
      }
      if (!fe.isStyleNameLoaded((n = o.$style) === null || n === void 0 ? void 0 : n.name) && (r = o.$style) !== null && r !== void 0 && r.name) {
        var h, _, E, w, P = ((h = o.$style) === null || h === void 0 || (_ = h.getDirectiveTheme) === null || _ === void 0 ? void 0 : _.call(h)) || {}, M = P.css, S = P.style;
        (E = o.$style) === null || E === void 0 || E.load(M, ie({
          name: "".concat(o.$style.name, "-variables")
        }, i)), (w = o.$style) === null || w === void 0 || w.loadStyle(ie({
          name: "".concat(o.$style.name, "-style")
        }, i), S), fe.setLoadedStyleName(o.$style.name);
      }
      if (!fe.isStyleNameLoaded("layer-order")) {
        var g, O, L = (g = o.$style) === null || g === void 0 || (O = g.getLayerOrderThemeCSS) === null || O === void 0 ? void 0 : O.call(g);
        ye.load(L, ie({
          name: "layer-order",
          first: !0
        }, i)), fe.setLoadedStyleName("layer-order");
      }
    }
  },
  _loadScopedThemeStyles: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, n = arguments.length > 1 ? arguments[1] : void 0, r = t.preset();
    if (r && t.$attrSelector) {
      var o, i, s, a = ((o = t.$style) === null || o === void 0 || (i = o.getPresetTheme) === null || i === void 0 ? void 0 : i.call(o, r, "[".concat(t.$attrSelector, "]"))) || {}, l = a.css, u = (s = t.$style) === null || s === void 0 ? void 0 : s.load(l, ie({
        name: "".concat(t.$attrSelector, "-").concat(t.$style.name)
      }, n));
      t.scopedStyleEl = u.el;
    }
  },
  _themeChangeListener: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : function() {
    };
    sn.clearLoadedStyleNames(), xe.on("theme:change", t);
  },
  _removeThemeListeners: function() {
    var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    xe.off("theme:change", t.$loadStyles), t.$loadStyles = void 0;
  },
  _hook: function(t, n, r, o, i, s) {
    var a, l, u = "on".concat(_p(n)), c = Z._getConfig(o, i), d = r == null ? void 0 : r.$instance, f = Z._usePT(d, Z._getPT(o == null || (a = o.value) === null || a === void 0 ? void 0 : a.pt, t), Z._getOptionValue, "hooks.".concat(u)), h = Z._useDefaultPT(d, c == null || (l = c.pt) === null || l === void 0 || (l = l.directives) === null || l === void 0 ? void 0 : l[t], Z._getOptionValue, "hooks.".concat(u)), _ = {
      el: r,
      binding: o,
      vnode: i,
      prevVnode: s
    };
    f == null || f(d, _), h == null || h(d, _);
  },
  /* eslint-disable-next-line no-unused-vars */
  _mergeProps: function() {
    for (var t = arguments.length > 1 ? arguments[1] : void 0, n = arguments.length, r = new Array(n > 2 ? n - 2 : 0), o = 2; o < n; o++)
      r[o - 2] = arguments[o];
    return es(t) ? t.apply(void 0, r) : de.apply(void 0, r);
  },
  _extend: function(t) {
    var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, r = function(a, l, u, c, d) {
      var f, h, _, E;
      l._$instances = l._$instances || {};
      var w = Z._getConfig(u, c), P = l._$instances[t] || {}, M = kn(P) ? ie(ie({}, n), n == null ? void 0 : n.methods) : {};
      l._$instances[t] = ie(ie({}, P), {}, {
        /* new instance variables to pass in directive methods */
        $name: t,
        $host: l,
        $binding: u,
        $modifiers: u == null ? void 0 : u.modifiers,
        $value: u == null ? void 0 : u.value,
        $el: P.$el || l || void 0,
        $style: ie({
          classes: void 0,
          inlineStyles: void 0,
          load: function() {
          },
          loadCSS: function() {
          },
          loadStyle: function() {
          }
        }, n == null ? void 0 : n.style),
        $primevueConfig: w,
        $attrSelector: (f = l.$pd) === null || f === void 0 || (f = f[t]) === null || f === void 0 ? void 0 : f.attrSelector,
        /* computed instance variables */
        defaultPT: function() {
          return Z._getPT(w == null ? void 0 : w.pt, void 0, function(g) {
            var O;
            return g == null || (O = g.directives) === null || O === void 0 ? void 0 : O[t];
          });
        },
        isUnstyled: function() {
          var g, O;
          return ((g = l._$instances[t]) === null || g === void 0 || (g = g.$binding) === null || g === void 0 || (g = g.value) === null || g === void 0 ? void 0 : g.unstyled) !== void 0 ? (O = l._$instances[t]) === null || O === void 0 || (O = O.$binding) === null || O === void 0 || (O = O.value) === null || O === void 0 ? void 0 : O.unstyled : w == null ? void 0 : w.unstyled;
        },
        theme: function() {
          var g;
          return (g = l._$instances[t]) === null || g === void 0 || (g = g.$primevueConfig) === null || g === void 0 ? void 0 : g.theme;
        },
        preset: function() {
          var g;
          return (g = l._$instances[t]) === null || g === void 0 || (g = g.$binding) === null || g === void 0 || (g = g.value) === null || g === void 0 ? void 0 : g.dt;
        },
        /* instance's methods */
        ptm: function() {
          var g, O = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", L = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
          return Z._getPTValue(l._$instances[t], (g = l._$instances[t]) === null || g === void 0 || (g = g.$binding) === null || g === void 0 || (g = g.value) === null || g === void 0 ? void 0 : g.pt, O, ie({}, L));
        },
        ptmo: function() {
          var g = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, O = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", L = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
          return Z._getPTValue(l._$instances[t], g, O, L, !1);
        },
        cx: function() {
          var g, O, L = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", A = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
          return (g = l._$instances[t]) !== null && g !== void 0 && g.isUnstyled() ? void 0 : Z._getOptionValue((O = l._$instances[t]) === null || O === void 0 || (O = O.$style) === null || O === void 0 ? void 0 : O.classes, L, ie({}, A));
        },
        sx: function() {
          var g, O = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : "", L = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !0, A = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
          return L ? Z._getOptionValue((g = l._$instances[t]) === null || g === void 0 || (g = g.$style) === null || g === void 0 ? void 0 : g.inlineStyles, O, ie({}, A)) : void 0;
        }
      }, M), l.$instance = l._$instances[t], (h = (_ = l.$instance)[a]) === null || h === void 0 || h.call(_, l, u, c, d), l["$".concat(t)] = l.$instance, Z._hook(t, a, l, u, c, d), l.$pd || (l.$pd = {}), l.$pd[t] = ie(ie({}, (E = l.$pd) === null || E === void 0 ? void 0 : E[t]), {}, {
        name: t,
        instance: l._$instances[t]
      });
    }, o = function(a) {
      var l, u, c, d = a._$instances[t], f = d == null ? void 0 : d.watch, h = function(w) {
        var P, M = w.newValue, S = w.oldValue;
        return f == null || (P = f.config) === null || P === void 0 ? void 0 : P.call(d, M, S);
      }, _ = function(w) {
        var P, M = w.newValue, S = w.oldValue;
        return f == null || (P = f["config.ripple"]) === null || P === void 0 ? void 0 : P.call(d, M, S);
      };
      d.$watchersCallback = {
        config: h,
        "config.ripple": _
      }, f == null || (l = f.config) === null || l === void 0 || l.call(d, d == null ? void 0 : d.$primevueConfig), an.on("config:change", h), f == null || (u = f["config.ripple"]) === null || u === void 0 || u.call(d, d == null || (c = d.$primevueConfig) === null || c === void 0 ? void 0 : c.ripple), an.on("config:ripple:change", _);
    }, i = function(a) {
      var l = a._$instances[t].$watchersCallback;
      l && (an.off("config:change", l.config), an.off("config:ripple:change", l["config.ripple"]), a._$instances[t].$watchersCallback = void 0);
    };
    return {
      created: function(a, l, u, c) {
        a.$pd || (a.$pd = {}), a.$pd[t] = {
          name: t,
          attrSelector: xp("pd")
        }, r("created", a, l, u, c);
      },
      beforeMount: function(a, l, u, c) {
        var d;
        Z._loadStyles((d = a.$pd[t]) === null || d === void 0 ? void 0 : d.instance, l, u), r("beforeMount", a, l, u, c), o(a);
      },
      mounted: function(a, l, u, c) {
        var d;
        Z._loadStyles((d = a.$pd[t]) === null || d === void 0 ? void 0 : d.instance, l, u), r("mounted", a, l, u, c);
      },
      beforeUpdate: function(a, l, u, c) {
        r("beforeUpdate", a, l, u, c);
      },
      updated: function(a, l, u, c) {
        var d;
        Z._loadStyles((d = a.$pd[t]) === null || d === void 0 ? void 0 : d.instance, l, u), r("updated", a, l, u, c);
      },
      beforeUnmount: function(a, l, u, c) {
        var d;
        i(a), Z._removeThemeListeners((d = a.$pd[t]) === null || d === void 0 ? void 0 : d.instance), r("beforeUnmount", a, l, u, c);
      },
      unmounted: function(a, l, u, c) {
        var d;
        (d = a.$pd[t]) === null || d === void 0 || (d = d.instance) === null || d === void 0 || (d = d.scopedStyleEl) === null || d === void 0 || (d = d.value) === null || d === void 0 || d.remove(), r("unmounted", a, l, u, c);
      }
    };
  },
  extend: function() {
    var t = Z._getMeta.apply(Z, arguments), n = Ea(t, 2), r = n[0], o = n[1];
    return ie({
      extend: function() {
        var s = Z._getMeta.apply(Z, arguments), a = Ea(s, 2), l = a[0], u = a[1];
        return Z.extend(l, ie(ie(ie({}, o), o == null ? void 0 : o.methods), u));
      }
    }, Z._extend(r, o));
  }
}, Ym = `
    .p-ink {
        display: block;
        position: absolute;
        background: dt('ripple.background');
        border-radius: 100%;
        transform: scale(0);
        pointer-events: none;
    }

    .p-ink-active {
        animation: ripple 0.4s linear;
    }

    @keyframes ripple {
        100% {
            opacity: 0;
            transform: scale(2.5);
        }
    }
`, zm = {
  root: "p-ink"
}, Gm = ye.extend({
  name: "ripple-directive",
  style: Ym,
  classes: zm
}), Xm = Z.extend({
  style: Gm
});
function Nr(e) {
  "@babel/helpers - typeof";
  return Nr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Nr(e);
}
function Jm(e) {
  return eh(e) || Qm(e) || Zm(e) || qm();
}
function qm() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Zm(e, t) {
  if (e) {
    if (typeof e == "string") return yi(e, t);
    var n = {}.toString.call(e).slice(8, -1);
    return n === "Object" && e.constructor && (n = e.constructor.name), n === "Map" || n === "Set" ? Array.from(e) : n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n) ? yi(e, t) : void 0;
  }
}
function Qm(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function eh(e) {
  if (Array.isArray(e)) return yi(e);
}
function yi(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var n = 0, r = Array(t); n < t; n++) r[n] = e[n];
  return r;
}
function Oa(e, t, n) {
  return (t = th(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function th(e) {
  var t = nh(e, "string");
  return Nr(t) == "symbol" ? t : t + "";
}
function nh(e, t) {
  if (Nr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Nr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var rc = Xm.extend("ripple", {
  watch: {
    "config.ripple": function(t) {
      t ? (this.createRipple(this.$host), this.bindEvents(this.$host), this.$host.setAttribute("data-pd-ripple", !0), this.$host.style.overflow = "hidden", this.$host.style.position = "relative") : (this.remove(this.$host), this.$host.removeAttribute("data-pd-ripple"));
    }
  },
  unmounted: function(t) {
    this.remove(t);
  },
  timeout: void 0,
  methods: {
    bindEvents: function(t) {
      t.addEventListener("mousedown", this.onMouseDown.bind(this));
    },
    unbindEvents: function(t) {
      t.removeEventListener("mousedown", this.onMouseDown.bind(this));
    },
    createRipple: function(t) {
      var n = this.getInk(t);
      n || (n = Ru("span", Oa(Oa({
        role: "presentation",
        "aria-hidden": !0,
        "data-p-ink": !0,
        "data-p-ink-active": !1,
        class: !this.isUnstyled() && this.cx("root"),
        onAnimationEnd: this.onAnimationEnd.bind(this)
      }, this.$attrSelector, ""), "p-bind", this.ptm("root"))), t.appendChild(n), this.$el = n);
    },
    remove: function(t) {
      var n = this.getInk(t);
      n && (this.$host.style.overflow = "", this.$host.style.position = "", this.unbindEvents(t), n.removeEventListener("animationend", this.onAnimationEnd), n.remove());
    },
    onMouseDown: function(t) {
      var n = this, r = t.currentTarget, o = this.getInk(r);
      if (!(!o || getComputedStyle(o, null).display === "none")) {
        if (!this.isUnstyled() && pr(o, "p-ink-active"), o.setAttribute("data-p-ink-active", "false"), !ta(o) && !na(o)) {
          var i = Math.max(Du(r), Fu(r));
          o.style.height = i + "px", o.style.width = i + "px";
        }
        var s = Ap(r), a = t.pageX - s.left + document.body.scrollTop - na(o) / 2, l = t.pageY - s.top + document.body.scrollLeft - ta(o) / 2;
        o.style.top = l + "px", o.style.left = a + "px", !this.isUnstyled() && po(o, "p-ink-active"), o.setAttribute("data-p-ink-active", "true"), this.timeout = setTimeout(function() {
          o && (!n.isUnstyled() && pr(o, "p-ink-active"), o.setAttribute("data-p-ink-active", "false"));
        }, 401);
      }
    },
    onAnimationEnd: function(t) {
      this.timeout && clearTimeout(this.timeout), !this.isUnstyled() && pr(t.currentTarget, "p-ink-active"), t.currentTarget.setAttribute("data-p-ink-active", "false");
    },
    getInk: function(t) {
      return t && t.children ? Jm(t.children).find(function(n) {
        return Np(n, "data-pc-name") === "ripple";
      }) : void 0;
    }
  }
}), rh = `
    .p-button {
        display: inline-flex;
        cursor: pointer;
        user-select: none;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        position: relative;
        color: dt('button.primary.color');
        background: dt('button.primary.background');
        border: 1px solid dt('button.primary.border.color');
        padding: dt('button.padding.y') dt('button.padding.x');
        font-size: 1rem;
        font-family: inherit;
        font-feature-settings: inherit;
        transition:
            background dt('button.transition.duration'),
            color dt('button.transition.duration'),
            border-color dt('button.transition.duration'),
            outline-color dt('button.transition.duration'),
            box-shadow dt('button.transition.duration');
        border-radius: dt('button.border.radius');
        outline-color: transparent;
        gap: dt('button.gap');
    }

    .p-button:disabled {
        cursor: default;
    }

    .p-button-icon-right {
        order: 1;
    }

    .p-button-icon-right:dir(rtl) {
        order: -1;
    }

    .p-button:not(.p-button-vertical) .p-button-icon:not(.p-button-icon-right):dir(rtl) {
        order: 1;
    }

    .p-button-icon-bottom {
        order: 2;
    }

    .p-button-icon-only {
        width: dt('button.icon.only.width');
        padding-inline-start: 0;
        padding-inline-end: 0;
        gap: 0;
    }

    .p-button-icon-only.p-button-rounded {
        border-radius: 50%;
        height: dt('button.icon.only.width');
    }

    .p-button-icon-only .p-button-label {
        visibility: hidden;
        width: 0;
    }

    .p-button-icon-only::after {
        content: "\0A0";
        visibility: hidden;
        width: 0;
    }

    .p-button-sm {
        font-size: dt('button.sm.font.size');
        padding: dt('button.sm.padding.y') dt('button.sm.padding.x');
    }

    .p-button-sm .p-button-icon {
        font-size: dt('button.sm.font.size');
    }

    .p-button-sm.p-button-icon-only {
        width: dt('button.sm.icon.only.width');
    }

    .p-button-sm.p-button-icon-only.p-button-rounded {
        height: dt('button.sm.icon.only.width');
    }

    .p-button-lg {
        font-size: dt('button.lg.font.size');
        padding: dt('button.lg.padding.y') dt('button.lg.padding.x');
    }

    .p-button-lg .p-button-icon {
        font-size: dt('button.lg.font.size');
    }

    .p-button-lg.p-button-icon-only {
        width: dt('button.lg.icon.only.width');
    }

    .p-button-lg.p-button-icon-only.p-button-rounded {
        height: dt('button.lg.icon.only.width');
    }

    .p-button-vertical {
        flex-direction: column;
    }

    .p-button-label {
        font-weight: dt('button.label.font.weight');
    }

    .p-button-fluid {
        width: 100%;
    }

    .p-button-fluid.p-button-icon-only {
        width: dt('button.icon.only.width');
    }

    .p-button:not(:disabled):hover {
        background: dt('button.primary.hover.background');
        border: 1px solid dt('button.primary.hover.border.color');
        color: dt('button.primary.hover.color');
    }

    .p-button:not(:disabled):active {
        background: dt('button.primary.active.background');
        border: 1px solid dt('button.primary.active.border.color');
        color: dt('button.primary.active.color');
    }

    .p-button:focus-visible {
        box-shadow: dt('button.primary.focus.ring.shadow');
        outline: dt('button.focus.ring.width') dt('button.focus.ring.style') dt('button.primary.focus.ring.color');
        outline-offset: dt('button.focus.ring.offset');
    }

    .p-button .p-badge {
        min-width: dt('button.badge.size');
        height: dt('button.badge.size');
        line-height: dt('button.badge.size');
    }

    .p-button-raised {
        box-shadow: dt('button.raised.shadow');
    }

    .p-button-rounded {
        border-radius: dt('button.rounded.border.radius');
    }

    .p-button-secondary {
        background: dt('button.secondary.background');
        border: 1px solid dt('button.secondary.border.color');
        color: dt('button.secondary.color');
    }

    .p-button-secondary:not(:disabled):hover {
        background: dt('button.secondary.hover.background');
        border: 1px solid dt('button.secondary.hover.border.color');
        color: dt('button.secondary.hover.color');
    }

    .p-button-secondary:not(:disabled):active {
        background: dt('button.secondary.active.background');
        border: 1px solid dt('button.secondary.active.border.color');
        color: dt('button.secondary.active.color');
    }

    .p-button-secondary:focus-visible {
        outline-color: dt('button.secondary.focus.ring.color');
        box-shadow: dt('button.secondary.focus.ring.shadow');
    }

    .p-button-success {
        background: dt('button.success.background');
        border: 1px solid dt('button.success.border.color');
        color: dt('button.success.color');
    }

    .p-button-success:not(:disabled):hover {
        background: dt('button.success.hover.background');
        border: 1px solid dt('button.success.hover.border.color');
        color: dt('button.success.hover.color');
    }

    .p-button-success:not(:disabled):active {
        background: dt('button.success.active.background');
        border: 1px solid dt('button.success.active.border.color');
        color: dt('button.success.active.color');
    }

    .p-button-success:focus-visible {
        outline-color: dt('button.success.focus.ring.color');
        box-shadow: dt('button.success.focus.ring.shadow');
    }

    .p-button-info {
        background: dt('button.info.background');
        border: 1px solid dt('button.info.border.color');
        color: dt('button.info.color');
    }

    .p-button-info:not(:disabled):hover {
        background: dt('button.info.hover.background');
        border: 1px solid dt('button.info.hover.border.color');
        color: dt('button.info.hover.color');
    }

    .p-button-info:not(:disabled):active {
        background: dt('button.info.active.background');
        border: 1px solid dt('button.info.active.border.color');
        color: dt('button.info.active.color');
    }

    .p-button-info:focus-visible {
        outline-color: dt('button.info.focus.ring.color');
        box-shadow: dt('button.info.focus.ring.shadow');
    }

    .p-button-warn {
        background: dt('button.warn.background');
        border: 1px solid dt('button.warn.border.color');
        color: dt('button.warn.color');
    }

    .p-button-warn:not(:disabled):hover {
        background: dt('button.warn.hover.background');
        border: 1px solid dt('button.warn.hover.border.color');
        color: dt('button.warn.hover.color');
    }

    .p-button-warn:not(:disabled):active {
        background: dt('button.warn.active.background');
        border: 1px solid dt('button.warn.active.border.color');
        color: dt('button.warn.active.color');
    }

    .p-button-warn:focus-visible {
        outline-color: dt('button.warn.focus.ring.color');
        box-shadow: dt('button.warn.focus.ring.shadow');
    }

    .p-button-help {
        background: dt('button.help.background');
        border: 1px solid dt('button.help.border.color');
        color: dt('button.help.color');
    }

    .p-button-help:not(:disabled):hover {
        background: dt('button.help.hover.background');
        border: 1px solid dt('button.help.hover.border.color');
        color: dt('button.help.hover.color');
    }

    .p-button-help:not(:disabled):active {
        background: dt('button.help.active.background');
        border: 1px solid dt('button.help.active.border.color');
        color: dt('button.help.active.color');
    }

    .p-button-help:focus-visible {
        outline-color: dt('button.help.focus.ring.color');
        box-shadow: dt('button.help.focus.ring.shadow');
    }

    .p-button-danger {
        background: dt('button.danger.background');
        border: 1px solid dt('button.danger.border.color');
        color: dt('button.danger.color');
    }

    .p-button-danger:not(:disabled):hover {
        background: dt('button.danger.hover.background');
        border: 1px solid dt('button.danger.hover.border.color');
        color: dt('button.danger.hover.color');
    }

    .p-button-danger:not(:disabled):active {
        background: dt('button.danger.active.background');
        border: 1px solid dt('button.danger.active.border.color');
        color: dt('button.danger.active.color');
    }

    .p-button-danger:focus-visible {
        outline-color: dt('button.danger.focus.ring.color');
        box-shadow: dt('button.danger.focus.ring.shadow');
    }

    .p-button-contrast {
        background: dt('button.contrast.background');
        border: 1px solid dt('button.contrast.border.color');
        color: dt('button.contrast.color');
    }

    .p-button-contrast:not(:disabled):hover {
        background: dt('button.contrast.hover.background');
        border: 1px solid dt('button.contrast.hover.border.color');
        color: dt('button.contrast.hover.color');
    }

    .p-button-contrast:not(:disabled):active {
        background: dt('button.contrast.active.background');
        border: 1px solid dt('button.contrast.active.border.color');
        color: dt('button.contrast.active.color');
    }

    .p-button-contrast:focus-visible {
        outline-color: dt('button.contrast.focus.ring.color');
        box-shadow: dt('button.contrast.focus.ring.shadow');
    }

    .p-button-outlined {
        background: transparent;
        border-color: dt('button.outlined.primary.border.color');
        color: dt('button.outlined.primary.color');
    }

    .p-button-outlined:not(:disabled):hover {
        background: dt('button.outlined.primary.hover.background');
        border-color: dt('button.outlined.primary.border.color');
        color: dt('button.outlined.primary.color');
    }

    .p-button-outlined:not(:disabled):active {
        background: dt('button.outlined.primary.active.background');
        border-color: dt('button.outlined.primary.border.color');
        color: dt('button.outlined.primary.color');
    }

    .p-button-outlined.p-button-secondary {
        border-color: dt('button.outlined.secondary.border.color');
        color: dt('button.outlined.secondary.color');
    }

    .p-button-outlined.p-button-secondary:not(:disabled):hover {
        background: dt('button.outlined.secondary.hover.background');
        border-color: dt('button.outlined.secondary.border.color');
        color: dt('button.outlined.secondary.color');
    }

    .p-button-outlined.p-button-secondary:not(:disabled):active {
        background: dt('button.outlined.secondary.active.background');
        border-color: dt('button.outlined.secondary.border.color');
        color: dt('button.outlined.secondary.color');
    }

    .p-button-outlined.p-button-success {
        border-color: dt('button.outlined.success.border.color');
        color: dt('button.outlined.success.color');
    }

    .p-button-outlined.p-button-success:not(:disabled):hover {
        background: dt('button.outlined.success.hover.background');
        border-color: dt('button.outlined.success.border.color');
        color: dt('button.outlined.success.color');
    }

    .p-button-outlined.p-button-success:not(:disabled):active {
        background: dt('button.outlined.success.active.background');
        border-color: dt('button.outlined.success.border.color');
        color: dt('button.outlined.success.color');
    }

    .p-button-outlined.p-button-info {
        border-color: dt('button.outlined.info.border.color');
        color: dt('button.outlined.info.color');
    }

    .p-button-outlined.p-button-info:not(:disabled):hover {
        background: dt('button.outlined.info.hover.background');
        border-color: dt('button.outlined.info.border.color');
        color: dt('button.outlined.info.color');
    }

    .p-button-outlined.p-button-info:not(:disabled):active {
        background: dt('button.outlined.info.active.background');
        border-color: dt('button.outlined.info.border.color');
        color: dt('button.outlined.info.color');
    }

    .p-button-outlined.p-button-warn {
        border-color: dt('button.outlined.warn.border.color');
        color: dt('button.outlined.warn.color');
    }

    .p-button-outlined.p-button-warn:not(:disabled):hover {
        background: dt('button.outlined.warn.hover.background');
        border-color: dt('button.outlined.warn.border.color');
        color: dt('button.outlined.warn.color');
    }

    .p-button-outlined.p-button-warn:not(:disabled):active {
        background: dt('button.outlined.warn.active.background');
        border-color: dt('button.outlined.warn.border.color');
        color: dt('button.outlined.warn.color');
    }

    .p-button-outlined.p-button-help {
        border-color: dt('button.outlined.help.border.color');
        color: dt('button.outlined.help.color');
    }

    .p-button-outlined.p-button-help:not(:disabled):hover {
        background: dt('button.outlined.help.hover.background');
        border-color: dt('button.outlined.help.border.color');
        color: dt('button.outlined.help.color');
    }

    .p-button-outlined.p-button-help:not(:disabled):active {
        background: dt('button.outlined.help.active.background');
        border-color: dt('button.outlined.help.border.color');
        color: dt('button.outlined.help.color');
    }

    .p-button-outlined.p-button-danger {
        border-color: dt('button.outlined.danger.border.color');
        color: dt('button.outlined.danger.color');
    }

    .p-button-outlined.p-button-danger:not(:disabled):hover {
        background: dt('button.outlined.danger.hover.background');
        border-color: dt('button.outlined.danger.border.color');
        color: dt('button.outlined.danger.color');
    }

    .p-button-outlined.p-button-danger:not(:disabled):active {
        background: dt('button.outlined.danger.active.background');
        border-color: dt('button.outlined.danger.border.color');
        color: dt('button.outlined.danger.color');
    }

    .p-button-outlined.p-button-contrast {
        border-color: dt('button.outlined.contrast.border.color');
        color: dt('button.outlined.contrast.color');
    }

    .p-button-outlined.p-button-contrast:not(:disabled):hover {
        background: dt('button.outlined.contrast.hover.background');
        border-color: dt('button.outlined.contrast.border.color');
        color: dt('button.outlined.contrast.color');
    }

    .p-button-outlined.p-button-contrast:not(:disabled):active {
        background: dt('button.outlined.contrast.active.background');
        border-color: dt('button.outlined.contrast.border.color');
        color: dt('button.outlined.contrast.color');
    }

    .p-button-outlined.p-button-plain {
        border-color: dt('button.outlined.plain.border.color');
        color: dt('button.outlined.plain.color');
    }

    .p-button-outlined.p-button-plain:not(:disabled):hover {
        background: dt('button.outlined.plain.hover.background');
        border-color: dt('button.outlined.plain.border.color');
        color: dt('button.outlined.plain.color');
    }

    .p-button-outlined.p-button-plain:not(:disabled):active {
        background: dt('button.outlined.plain.active.background');
        border-color: dt('button.outlined.plain.border.color');
        color: dt('button.outlined.plain.color');
    }

    .p-button-text {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.primary.color');
    }

    .p-button-text:not(:disabled):hover {
        background: dt('button.text.primary.hover.background');
        border-color: transparent;
        color: dt('button.text.primary.color');
    }

    .p-button-text:not(:disabled):active {
        background: dt('button.text.primary.active.background');
        border-color: transparent;
        color: dt('button.text.primary.color');
    }

    .p-button-text.p-button-secondary {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.secondary.color');
    }

    .p-button-text.p-button-secondary:not(:disabled):hover {
        background: dt('button.text.secondary.hover.background');
        border-color: transparent;
        color: dt('button.text.secondary.color');
    }

    .p-button-text.p-button-secondary:not(:disabled):active {
        background: dt('button.text.secondary.active.background');
        border-color: transparent;
        color: dt('button.text.secondary.color');
    }

    .p-button-text.p-button-success {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.success.color');
    }

    .p-button-text.p-button-success:not(:disabled):hover {
        background: dt('button.text.success.hover.background');
        border-color: transparent;
        color: dt('button.text.success.color');
    }

    .p-button-text.p-button-success:not(:disabled):active {
        background: dt('button.text.success.active.background');
        border-color: transparent;
        color: dt('button.text.success.color');
    }

    .p-button-text.p-button-info {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.info.color');
    }

    .p-button-text.p-button-info:not(:disabled):hover {
        background: dt('button.text.info.hover.background');
        border-color: transparent;
        color: dt('button.text.info.color');
    }

    .p-button-text.p-button-info:not(:disabled):active {
        background: dt('button.text.info.active.background');
        border-color: transparent;
        color: dt('button.text.info.color');
    }

    .p-button-text.p-button-warn {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.warn.color');
    }

    .p-button-text.p-button-warn:not(:disabled):hover {
        background: dt('button.text.warn.hover.background');
        border-color: transparent;
        color: dt('button.text.warn.color');
    }

    .p-button-text.p-button-warn:not(:disabled):active {
        background: dt('button.text.warn.active.background');
        border-color: transparent;
        color: dt('button.text.warn.color');
    }

    .p-button-text.p-button-help {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.help.color');
    }

    .p-button-text.p-button-help:not(:disabled):hover {
        background: dt('button.text.help.hover.background');
        border-color: transparent;
        color: dt('button.text.help.color');
    }

    .p-button-text.p-button-help:not(:disabled):active {
        background: dt('button.text.help.active.background');
        border-color: transparent;
        color: dt('button.text.help.color');
    }

    .p-button-text.p-button-danger {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.danger.color');
    }

    .p-button-text.p-button-danger:not(:disabled):hover {
        background: dt('button.text.danger.hover.background');
        border-color: transparent;
        color: dt('button.text.danger.color');
    }

    .p-button-text.p-button-danger:not(:disabled):active {
        background: dt('button.text.danger.active.background');
        border-color: transparent;
        color: dt('button.text.danger.color');
    }

    .p-button-text.p-button-contrast {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.contrast.color');
    }

    .p-button-text.p-button-contrast:not(:disabled):hover {
        background: dt('button.text.contrast.hover.background');
        border-color: transparent;
        color: dt('button.text.contrast.color');
    }

    .p-button-text.p-button-contrast:not(:disabled):active {
        background: dt('button.text.contrast.active.background');
        border-color: transparent;
        color: dt('button.text.contrast.color');
    }

    .p-button-text.p-button-plain {
        background: transparent;
        border-color: transparent;
        color: dt('button.text.plain.color');
    }

    .p-button-text.p-button-plain:not(:disabled):hover {
        background: dt('button.text.plain.hover.background');
        border-color: transparent;
        color: dt('button.text.plain.color');
    }

    .p-button-text.p-button-plain:not(:disabled):active {
        background: dt('button.text.plain.active.background');
        border-color: transparent;
        color: dt('button.text.plain.color');
    }

    .p-button-link {
        background: transparent;
        border-color: transparent;
        color: dt('button.link.color');
    }

    .p-button-link:not(:disabled):hover {
        background: transparent;
        border-color: transparent;
        color: dt('button.link.hover.color');
    }

    .p-button-link:not(:disabled):hover .p-button-label {
        text-decoration: underline;
    }

    .p-button-link:not(:disabled):active {
        background: transparent;
        border-color: transparent;
        color: dt('button.link.active.color');
    }
`;
function Ir(e) {
  "@babel/helpers - typeof";
  return Ir = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Ir(e);
}
function $t(e, t, n) {
  return (t = oh(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function oh(e) {
  var t = ih(e, "string");
  return Ir(t) == "symbol" ? t : t + "";
}
function ih(e, t) {
  if (Ir(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Ir(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var sh = {
  root: function(t) {
    var n = t.instance, r = t.props;
    return ["p-button p-component", $t($t($t($t($t($t($t($t($t({
      "p-button-icon-only": n.hasIcon && !r.label && !r.badge,
      "p-button-vertical": (r.iconPos === "top" || r.iconPos === "bottom") && r.label,
      "p-button-loading": r.loading,
      "p-button-link": r.link || r.variant === "link"
    }, "p-button-".concat(r.severity), r.severity), "p-button-raised", r.raised), "p-button-rounded", r.rounded), "p-button-text", r.text || r.variant === "text"), "p-button-outlined", r.outlined || r.variant === "outlined"), "p-button-sm", r.size === "small"), "p-button-lg", r.size === "large"), "p-button-plain", r.plain), "p-button-fluid", n.hasFluid)];
  },
  loadingIcon: "p-button-loading-icon",
  icon: function(t) {
    var n = t.props;
    return ["p-button-icon", $t({}, "p-button-icon-".concat(n.iconPos), n.label)];
  },
  label: "p-button-label"
}, ah = ye.extend({
  name: "button",
  style: rh,
  classes: sh
}), lh = {
  name: "BaseButton",
  extends: ko,
  props: {
    label: {
      type: String,
      default: null
    },
    icon: {
      type: String,
      default: null
    },
    iconPos: {
      type: String,
      default: "left"
    },
    iconClass: {
      type: [String, Object],
      default: null
    },
    badge: {
      type: String,
      default: null
    },
    badgeClass: {
      type: [String, Object],
      default: null
    },
    badgeSeverity: {
      type: String,
      default: "secondary"
    },
    loading: {
      type: Boolean,
      default: !1
    },
    loadingIcon: {
      type: String,
      default: void 0
    },
    as: {
      type: [String, Object],
      default: "BUTTON"
    },
    asChild: {
      type: Boolean,
      default: !1
    },
    link: {
      type: Boolean,
      default: !1
    },
    severity: {
      type: String,
      default: null
    },
    raised: {
      type: Boolean,
      default: !1
    },
    rounded: {
      type: Boolean,
      default: !1
    },
    text: {
      type: Boolean,
      default: !1
    },
    outlined: {
      type: Boolean,
      default: !1
    },
    size: {
      type: String,
      default: null
    },
    variant: {
      type: String,
      default: null
    },
    plain: {
      type: Boolean,
      default: !1
    },
    fluid: {
      type: Boolean,
      default: null
    }
  },
  style: ah,
  provide: function() {
    return {
      $pcButton: this,
      $parentInstance: this
    };
  }
};
function Ar(e) {
  "@babel/helpers - typeof";
  return Ar = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Ar(e);
}
function tt(e, t, n) {
  return (t = uh(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function uh(e) {
  var t = ch(e, "string");
  return Ar(t) == "symbol" ? t : t + "";
}
function ch(e, t) {
  if (Ar(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Ar(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var oc = {
  name: "Button",
  extends: lh,
  inheritAttrs: !1,
  inject: {
    $pcFluid: {
      default: null
    }
  },
  methods: {
    getPTOptions: function(t) {
      var n = t === "root" ? this.ptmi : this.ptm;
      return n(t, {
        context: {
          disabled: this.disabled
        }
      });
    }
  },
  computed: {
    disabled: function() {
      return this.$attrs.disabled || this.$attrs.disabled === "" || this.loading;
    },
    defaultAriaLabel: function() {
      return this.label ? this.label + (this.badge ? " " + this.badge : "") : this.$attrs.ariaLabel;
    },
    hasIcon: function() {
      return this.icon || this.$slots.icon;
    },
    attrs: function() {
      return de(this.asAttrs, this.a11yAttrs, this.getPTOptions("root"));
    },
    asAttrs: function() {
      return this.as === "BUTTON" ? {
        type: "button",
        disabled: this.disabled
      } : void 0;
    },
    a11yAttrs: function() {
      return {
        "aria-label": this.defaultAriaLabel,
        "data-pc-name": "button",
        "data-p-disabled": this.disabled,
        "data-p-severity": this.severity
      };
    },
    hasFluid: function() {
      return kn(this.fluid) ? !!this.$pcFluid : this.fluid;
    },
    dataP: function() {
      return Wn(tt(tt(tt(tt(tt(tt(tt(tt(tt(tt({}, this.size, this.size), "icon-only", this.hasIcon && !this.label && !this.badge), "loading", this.loading), "fluid", this.hasFluid), "rounded", this.rounded), "raised", this.raised), "outlined", this.outlined || this.variant === "outlined"), "text", this.text || this.variant === "text"), "link", this.link || this.variant === "link"), "vertical", (this.iconPos === "top" || this.iconPos === "bottom") && this.label));
    },
    dataIconP: function() {
      return Wn(tt(tt({}, this.iconPos, this.iconPos), this.size, this.size));
    },
    dataLabelP: function() {
      return Wn(tt(tt({}, this.size, this.size), "icon-only", this.hasIcon && !this.label && !this.badge));
    }
  },
  components: {
    SpinnerIcon: tc,
    Badge: nc
  },
  directives: {
    ripple: rc
  }
}, dh = ["data-p"], fh = ["data-p"];
function ph(e, t, n, r, o, i) {
  var s = uo("SpinnerIcon"), a = uo("Badge"), l = au("ripple");
  return e.asChild ? Xe(e.$slots, "default", {
    key: 1,
    class: Bn(e.cx("root")),
    a11yAttrs: i.a11yAttrs
  }) : zl((be(), Ct(ui(e.as), de({
    key: 0,
    class: e.cx("root"),
    "data-p": i.dataP
  }, i.attrs), {
    default: Tn(function() {
      return [Xe(e.$slots, "default", {}, function() {
        return [e.loading ? Xe(e.$slots, "loadingicon", de({
          key: 0,
          class: [e.cx("loadingIcon"), e.cx("icon")]
        }, e.ptm("loadingIcon")), function() {
          return [e.loadingIcon ? (be(), Ze("span", de({
            key: 0,
            class: [e.cx("loadingIcon"), e.cx("icon"), e.loadingIcon]
          }, e.ptm("loadingIcon")), null, 16)) : (be(), Ct(s, de({
            key: 1,
            class: [e.cx("loadingIcon"), e.cx("icon")],
            spin: ""
          }, e.ptm("loadingIcon")), null, 16, ["class"]))];
        }) : Xe(e.$slots, "icon", de({
          key: 1,
          class: [e.cx("icon")]
        }, e.ptm("icon")), function() {
          return [e.icon ? (be(), Ze("span", de({
            key: 0,
            class: [e.cx("icon"), e.icon, e.iconClass],
            "data-p": i.dataIconP
          }, e.ptm("icon")), null, 16, dh)) : vt("", !0)];
        }), e.label ? (be(), Ze("span", de({
          key: 2,
          class: e.cx("label")
        }, e.ptm("label"), {
          "data-p": i.dataLabelP
        }), mr(e.label), 17, fh)) : vt("", !0), e.badge ? (be(), Ct(a, {
          key: 3,
          value: e.badge,
          class: Bn(e.badgeClass),
          severity: e.badgeSeverity,
          unstyled: e.unstyled,
          pt: e.ptm("pcBadge")
        }, null, 8, ["value", "class", "severity", "unstyled", "pt"])) : vt("", !0)];
      })];
    }),
    _: 3
  }, 16, ["class", "data-p"])), [[l]]);
}
oc.render = ph;
var mh = ye.extend({
  name: "focustrap-directive"
}), hh = Z.extend({
  style: mh
});
function xr(e) {
  "@babel/helpers - typeof";
  return xr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, xr(e);
}
function La(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function Pa(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? La(Object(n), !0).forEach(function(r) {
      gh(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : La(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function gh(e, t, n) {
  return (t = bh(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function bh(e) {
  var t = vh(e, "string");
  return xr(t) == "symbol" ? t : t + "";
}
function vh(e, t) {
  if (xr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (xr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var yh = hh.extend("focustrap", {
  mounted: function(t, n) {
    var r = n.value || {}, o = r.disabled;
    o || (this.createHiddenFocusableElements(t, n), this.bind(t, n), this.autoElementFocus(t, n)), t.setAttribute("data-pd-focustrap", !0), this.$el = t;
  },
  updated: function(t, n) {
    var r = n.value || {}, o = r.disabled;
    o && this.unbind(t);
  },
  unmounted: function(t) {
    this.unbind(t);
  },
  methods: {
    getComputedSelector: function(t) {
      return ':not(.p-hidden-focusable):not([data-p-hidden-focusable="true"])'.concat(t ?? "");
    },
    bind: function(t, n) {
      var r = this, o = n.value || {}, i = o.onFocusIn, s = o.onFocusOut;
      t.$_pfocustrap_mutationobserver = new MutationObserver(function(a) {
        a.forEach(function(l) {
          if (l.type === "childList" && !t.contains(document.activeElement)) {
            var u = function(d) {
              var f = ra(d) ? ra(d, r.getComputedSelector(t.$_pfocustrap_focusableselector)) ? d : Jn(t, r.getComputedSelector(t.$_pfocustrap_focusableselector)) : Jn(d);
              return _e(f) ? f : d.nextSibling && u(d.nextSibling);
            };
            Dn(u(l.nextSibling));
          }
        });
      }), t.$_pfocustrap_mutationobserver.disconnect(), t.$_pfocustrap_mutationobserver.observe(t, {
        childList: !0
      }), t.$_pfocustrap_focusinlistener = function(a) {
        return i && i(a);
      }, t.$_pfocustrap_focusoutlistener = function(a) {
        return s && s(a);
      }, t.addEventListener("focusin", t.$_pfocustrap_focusinlistener), t.addEventListener("focusout", t.$_pfocustrap_focusoutlistener);
    },
    unbind: function(t) {
      t.$_pfocustrap_mutationobserver && t.$_pfocustrap_mutationobserver.disconnect(), t.$_pfocustrap_focusinlistener && t.removeEventListener("focusin", t.$_pfocustrap_focusinlistener) && (t.$_pfocustrap_focusinlistener = null), t.$_pfocustrap_focusoutlistener && t.removeEventListener("focusout", t.$_pfocustrap_focusoutlistener) && (t.$_pfocustrap_focusoutlistener = null);
    },
    autoFocus: function(t) {
      this.autoElementFocus(this.$el, {
        value: Pa(Pa({}, t), {}, {
          autoFocus: !0
        })
      });
    },
    autoElementFocus: function(t, n) {
      var r = n.value || {}, o = r.autoFocusSelector, i = o === void 0 ? "" : o, s = r.firstFocusableSelector, a = s === void 0 ? "" : s, l = r.autoFocus, u = l === void 0 ? !1 : l, c = Jn(t, "[autofocus]".concat(this.getComputedSelector(i)));
      u && !c && (c = Jn(t, this.getComputedSelector(a))), Dn(c);
    },
    onFirstHiddenElementFocus: function(t) {
      var n, r = t.currentTarget, o = t.relatedTarget, i = o === r.$_pfocustrap_lasthiddenfocusableelement || !((n = this.$el) !== null && n !== void 0 && n.contains(o)) ? Jn(r.parentElement, this.getComputedSelector(r.$_pfocustrap_focusableselector)) : r.$_pfocustrap_lasthiddenfocusableelement;
      Dn(i);
    },
    onLastHiddenElementFocus: function(t) {
      var n, r = t.currentTarget, o = t.relatedTarget, i = o === r.$_pfocustrap_firsthiddenfocusableelement || !((n = this.$el) !== null && n !== void 0 && n.contains(o)) ? Ip(r.parentElement, this.getComputedSelector(r.$_pfocustrap_focusableselector)) : r.$_pfocustrap_firsthiddenfocusableelement;
      Dn(i);
    },
    createHiddenFocusableElements: function(t, n) {
      var r = this, o = n.value || {}, i = o.tabIndex, s = i === void 0 ? 0 : i, a = o.firstFocusableSelector, l = a === void 0 ? "" : a, u = o.lastFocusableSelector, c = u === void 0 ? "" : u, d = function(E) {
        return Ru("span", {
          class: "p-hidden-accessible p-hidden-focusable",
          tabIndex: s,
          role: "presentation",
          "aria-hidden": !0,
          "data-p-hidden-accessible": !0,
          "data-p-hidden-focusable": !0,
          onFocus: E == null ? void 0 : E.bind(r)
        });
      }, f = d(this.onFirstHiddenElementFocus), h = d(this.onLastHiddenElementFocus);
      f.$_pfocustrap_lasthiddenfocusableelement = h, f.$_pfocustrap_focusableselector = l, f.setAttribute("data-pc-section", "firstfocusableelement"), h.$_pfocustrap_firsthiddenfocusableelement = f, h.$_pfocustrap_focusableselector = c, h.setAttribute("data-pc-section", "lastfocusableelement"), t.prepend(f), t.append(h);
    }
  }
}), ic = {
  name: "Portal",
  props: {
    appendTo: {
      type: [String, Object],
      default: "body"
    },
    disabled: {
      type: Boolean,
      default: !1
    }
  },
  data: function() {
    return {
      mounted: !1
    };
  },
  mounted: function() {
    this.mounted = ju();
  },
  computed: {
    inline: function() {
      return this.disabled || this.appendTo === "self";
    }
  }
};
function _h(e, t, n, r, o, i) {
  return i.inline ? Xe(e.$slots, "default", {
    key: 0
  }) : o.mounted ? (be(), Ct($d, {
    key: 1,
    to: n.appendTo
  }, [Xe(e.$slots, "default")], 8, ["to"])) : vt("", !0);
}
ic.render = _h;
function wa() {
  Tp({
    variableName: Gu("scrollbar.width").name
  });
}
function ka() {
  Cp({
    variableName: Gu("scrollbar.width").name
  });
}
var Sh = `
    .p-dialog {
        max-height: 90%;
        transform: scale(1);
        border-radius: dt('dialog.border.radius');
        box-shadow: dt('dialog.shadow');
        background: dt('dialog.background');
        border: 1px solid dt('dialog.border.color');
        color: dt('dialog.color');
    }

    .p-dialog-content {
        overflow-y: auto;
        padding: dt('dialog.content.padding');
    }

    .p-dialog-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-shrink: 0;
        padding: dt('dialog.header.padding');
    }

    .p-dialog-title {
        font-weight: dt('dialog.title.font.weight');
        font-size: dt('dialog.title.font.size');
    }

    .p-dialog-footer {
        flex-shrink: 0;
        padding: dt('dialog.footer.padding');
        display: flex;
        justify-content: flex-end;
        gap: dt('dialog.footer.gap');
    }

    .p-dialog-header-actions {
        display: flex;
        align-items: center;
        gap: dt('dialog.header.gap');
    }

    .p-dialog-enter-active {
        transition: all 150ms cubic-bezier(0, 0, 0.2, 1);
    }

    .p-dialog-leave-active {
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .p-dialog-enter-from,
    .p-dialog-leave-to {
        opacity: 0;
        transform: scale(0.7);
    }

    .p-dialog-top .p-dialog,
    .p-dialog-bottom .p-dialog,
    .p-dialog-left .p-dialog,
    .p-dialog-right .p-dialog,
    .p-dialog-topleft .p-dialog,
    .p-dialog-topright .p-dialog,
    .p-dialog-bottomleft .p-dialog,
    .p-dialog-bottomright .p-dialog {
        margin: 0.75rem;
        transform: translate3d(0px, 0px, 0px);
    }

    .p-dialog-top .p-dialog-enter-active,
    .p-dialog-top .p-dialog-leave-active,
    .p-dialog-bottom .p-dialog-enter-active,
    .p-dialog-bottom .p-dialog-leave-active,
    .p-dialog-left .p-dialog-enter-active,
    .p-dialog-left .p-dialog-leave-active,
    .p-dialog-right .p-dialog-enter-active,
    .p-dialog-right .p-dialog-leave-active,
    .p-dialog-topleft .p-dialog-enter-active,
    .p-dialog-topleft .p-dialog-leave-active,
    .p-dialog-topright .p-dialog-enter-active,
    .p-dialog-topright .p-dialog-leave-active,
    .p-dialog-bottomleft .p-dialog-enter-active,
    .p-dialog-bottomleft .p-dialog-leave-active,
    .p-dialog-bottomright .p-dialog-enter-active,
    .p-dialog-bottomright .p-dialog-leave-active {
        transition: all 0.3s ease-out;
    }

    .p-dialog-top .p-dialog-enter-from,
    .p-dialog-top .p-dialog-leave-to {
        transform: translate3d(0px, -100%, 0px);
    }

    .p-dialog-bottom .p-dialog-enter-from,
    .p-dialog-bottom .p-dialog-leave-to {
        transform: translate3d(0px, 100%, 0px);
    }

    .p-dialog-left .p-dialog-enter-from,
    .p-dialog-left .p-dialog-leave-to,
    .p-dialog-topleft .p-dialog-enter-from,
    .p-dialog-topleft .p-dialog-leave-to,
    .p-dialog-bottomleft .p-dialog-enter-from,
    .p-dialog-bottomleft .p-dialog-leave-to {
        transform: translate3d(-100%, 0px, 0px);
    }

    .p-dialog-right .p-dialog-enter-from,
    .p-dialog-right .p-dialog-leave-to,
    .p-dialog-topright .p-dialog-enter-from,
    .p-dialog-topright .p-dialog-leave-to,
    .p-dialog-bottomright .p-dialog-enter-from,
    .p-dialog-bottomright .p-dialog-leave-to {
        transform: translate3d(100%, 0px, 0px);
    }

    .p-dialog-left:dir(rtl) .p-dialog-enter-from,
    .p-dialog-left:dir(rtl) .p-dialog-leave-to,
    .p-dialog-topleft:dir(rtl) .p-dialog-enter-from,
    .p-dialog-topleft:dir(rtl) .p-dialog-leave-to,
    .p-dialog-bottomleft:dir(rtl) .p-dialog-enter-from,
    .p-dialog-bottomleft:dir(rtl) .p-dialog-leave-to {
        transform: translate3d(100%, 0px, 0px);
    }

    .p-dialog-right:dir(rtl) .p-dialog-enter-from,
    .p-dialog-right:dir(rtl) .p-dialog-leave-to,
    .p-dialog-topright:dir(rtl) .p-dialog-enter-from,
    .p-dialog-topright:dir(rtl) .p-dialog-leave-to,
    .p-dialog-bottomright:dir(rtl) .p-dialog-enter-from,
    .p-dialog-bottomright:dir(rtl) .p-dialog-leave-to {
        transform: translate3d(-100%, 0px, 0px);
    }

    .p-dialog-maximized {
        width: 100vw !important;
        height: 100vh !important;
        top: 0px !important;
        left: 0px !important;
        max-height: 100%;
        height: 100%;
        border-radius: 0;
    }

    .p-dialog-maximized .p-dialog-content {
        flex-grow: 1;
    }

    .p-dialog .p-resizable-handle {
        position: absolute;
        font-size: 0.1px;
        display: block;
        cursor: se-resize;
        width: 12px;
        height: 12px;
        right: 1px;
        bottom: 1px;
    }
`, Eh = {
  mask: function(t) {
    var n = t.position, r = t.modal;
    return {
      position: "fixed",
      height: "100%",
      width: "100%",
      left: 0,
      top: 0,
      display: "flex",
      justifyContent: n === "left" || n === "topleft" || n === "bottomleft" ? "flex-start" : n === "right" || n === "topright" || n === "bottomright" ? "flex-end" : "center",
      alignItems: n === "top" || n === "topleft" || n === "topright" ? "flex-start" : n === "bottom" || n === "bottomleft" || n === "bottomright" ? "flex-end" : "center",
      pointerEvents: r ? "auto" : "none"
    };
  },
  root: {
    display: "flex",
    flexDirection: "column",
    pointerEvents: "auto"
  }
}, Th = {
  mask: function(t) {
    var n = t.props, r = ["left", "right", "top", "topleft", "topright", "bottom", "bottomleft", "bottomright"], o = r.find(function(i) {
      return i === n.position;
    });
    return ["p-dialog-mask", {
      "p-overlay-mask p-overlay-mask-enter": n.modal
    }, o ? "p-dialog-".concat(o) : ""];
  },
  root: function(t) {
    var n = t.props, r = t.instance;
    return ["p-dialog p-component", {
      "p-dialog-maximized": n.maximizable && r.maximized
    }];
  },
  header: "p-dialog-header",
  title: "p-dialog-title",
  headerActions: "p-dialog-header-actions",
  pcMaximizeButton: "p-dialog-maximize-button",
  pcCloseButton: "p-dialog-close-button",
  content: "p-dialog-content",
  footer: "p-dialog-footer"
}, Ch = ye.extend({
  name: "dialog",
  style: Sh,
  classes: Th,
  inlineStyles: Eh
}), Oh = {
  name: "BaseDialog",
  extends: ko,
  props: {
    header: {
      type: null,
      default: null
    },
    footer: {
      type: null,
      default: null
    },
    visible: {
      type: Boolean,
      default: !1
    },
    modal: {
      type: Boolean,
      default: null
    },
    contentStyle: {
      type: null,
      default: null
    },
    contentClass: {
      type: String,
      default: null
    },
    contentProps: {
      type: null,
      default: null
    },
    maximizable: {
      type: Boolean,
      default: !1
    },
    dismissableMask: {
      type: Boolean,
      default: !1
    },
    closable: {
      type: Boolean,
      default: !0
    },
    closeOnEscape: {
      type: Boolean,
      default: !0
    },
    showHeader: {
      type: Boolean,
      default: !0
    },
    blockScroll: {
      type: Boolean,
      default: !1
    },
    baseZIndex: {
      type: Number,
      default: 0
    },
    autoZIndex: {
      type: Boolean,
      default: !0
    },
    position: {
      type: String,
      default: "center"
    },
    breakpoints: {
      type: Object,
      default: null
    },
    draggable: {
      type: Boolean,
      default: !0
    },
    keepInViewport: {
      type: Boolean,
      default: !0
    },
    minX: {
      type: Number,
      default: 0
    },
    minY: {
      type: Number,
      default: 0
    },
    appendTo: {
      type: [String, Object],
      default: "body"
    },
    closeIcon: {
      type: String,
      default: void 0
    },
    maximizeIcon: {
      type: String,
      default: void 0
    },
    minimizeIcon: {
      type: String,
      default: void 0
    },
    closeButtonProps: {
      type: Object,
      default: function() {
        return {
          severity: "secondary",
          text: !0,
          rounded: !0
        };
      }
    },
    maximizeButtonProps: {
      type: Object,
      default: function() {
        return {
          severity: "secondary",
          text: !0,
          rounded: !0
        };
      }
    },
    _instance: null
  },
  style: Ch,
  provide: function() {
    return {
      $pcDialog: this,
      $parentInstance: this
    };
  }
}, sc = {
  name: "Dialog",
  extends: Oh,
  inheritAttrs: !1,
  emits: ["update:visible", "show", "hide", "after-hide", "maximize", "unmaximize", "dragstart", "dragend"],
  provide: function() {
    var t = this;
    return {
      dialogRef: dt(function() {
        return t._instance;
      })
    };
  },
  data: function() {
    return {
      containerVisible: this.visible,
      maximized: !1,
      focusableMax: null,
      focusableClose: null,
      target: null
    };
  },
  documentKeydownListener: null,
  container: null,
  mask: null,
  content: null,
  headerContainer: null,
  footerContainer: null,
  maximizableButton: null,
  closeButton: null,
  styleElement: null,
  dragging: null,
  documentDragListener: null,
  documentDragEndListener: null,
  lastPageX: null,
  lastPageY: null,
  maskMouseDownTarget: null,
  updated: function() {
    this.visible && (this.containerVisible = this.visible);
  },
  beforeUnmount: function() {
    this.unbindDocumentState(), this.unbindGlobalListeners(), this.destroyStyle(), this.mask && this.autoZIndex && Xo.clear(this.mask), this.container = null, this.mask = null;
  },
  mounted: function() {
    this.breakpoints && this.createStyle();
  },
  methods: {
    close: function() {
      this.$emit("update:visible", !1);
    },
    onEnter: function() {
      this.$emit("show"), this.target = document.activeElement, this.enableDocumentSettings(), this.bindGlobalListeners(), this.autoZIndex && Xo.set("modal", this.mask, this.baseZIndex + this.$primevue.config.zIndex.modal);
    },
    onAfterEnter: function() {
      this.focus();
    },
    onBeforeLeave: function() {
      this.modal && !this.isUnstyled && po(this.mask, "p-overlay-mask-leave"), this.dragging && this.documentDragEndListener && this.documentDragEndListener();
    },
    onLeave: function() {
      this.$emit("hide"), Dn(this.target), this.target = null, this.focusableClose = null, this.focusableMax = null;
    },
    onAfterLeave: function() {
      this.autoZIndex && Xo.clear(this.mask), this.containerVisible = !1, this.unbindDocumentState(), this.unbindGlobalListeners(), this.$emit("after-hide");
    },
    onMaskMouseDown: function(t) {
      this.maskMouseDownTarget = t.target;
    },
    onMaskMouseUp: function() {
      this.dismissableMask && this.modal && this.mask === this.maskMouseDownTarget && this.close();
    },
    focus: function() {
      var t = function(o) {
        return o && o.querySelector("[autofocus]");
      }, n = this.$slots.footer && t(this.footerContainer);
      n || (n = this.$slots.header && t(this.headerContainer), n || (n = this.$slots.default && t(this.content), n || (this.maximizable ? (this.focusableMax = !0, n = this.maximizableButton) : (this.focusableClose = !0, n = this.closeButton)))), n && Dn(n, {
        focusVisible: !0
      });
    },
    maximize: function(t) {
      this.maximized ? (this.maximized = !1, this.$emit("unmaximize", t)) : (this.maximized = !0, this.$emit("maximize", t)), this.modal || (this.maximized ? wa() : ka());
    },
    enableDocumentSettings: function() {
      (this.modal || !this.modal && this.blockScroll || this.maximizable && this.maximized) && wa();
    },
    unbindDocumentState: function() {
      (this.modal || !this.modal && this.blockScroll || this.maximizable && this.maximized) && ka();
    },
    onKeyDown: function(t) {
      t.code === "Escape" && this.closeOnEscape && this.close();
    },
    bindDocumentKeyDownListener: function() {
      this.documentKeydownListener || (this.documentKeydownListener = this.onKeyDown.bind(this), window.document.addEventListener("keydown", this.documentKeydownListener));
    },
    unbindDocumentKeyDownListener: function() {
      this.documentKeydownListener && (window.document.removeEventListener("keydown", this.documentKeydownListener), this.documentKeydownListener = null);
    },
    containerRef: function(t) {
      this.container = t;
    },
    maskRef: function(t) {
      this.mask = t;
    },
    contentRef: function(t) {
      this.content = t;
    },
    headerContainerRef: function(t) {
      this.headerContainer = t;
    },
    footerContainerRef: function(t) {
      this.footerContainer = t;
    },
    maximizableRef: function(t) {
      this.maximizableButton = t ? t.$el : void 0;
    },
    closeButtonRef: function(t) {
      this.closeButton = t ? t.$el : void 0;
    },
    createStyle: function() {
      if (!this.styleElement && !this.isUnstyled) {
        var t;
        this.styleElement = document.createElement("style"), this.styleElement.type = "text/css", Uu(this.styleElement, "nonce", (t = this.$primevue) === null || t === void 0 || (t = t.config) === null || t === void 0 || (t = t.csp) === null || t === void 0 ? void 0 : t.nonce), document.head.appendChild(this.styleElement);
        var n = "";
        for (var r in this.breakpoints)
          n += `
                        @media screen and (max-width: `.concat(r, `) {
                            .p-dialog[`).concat(this.$attrSelector, `] {
                                width: `).concat(this.breakpoints[r], ` !important;
                            }
                        }
                    `);
        this.styleElement.innerHTML = n;
      }
    },
    destroyStyle: function() {
      this.styleElement && (document.head.removeChild(this.styleElement), this.styleElement = null);
    },
    initDrag: function(t) {
      t.target.closest("div").getAttribute("data-pc-section") !== "headeractions" && this.draggable && (this.dragging = !0, this.lastPageX = t.pageX, this.lastPageY = t.pageY, this.container.style.margin = "0", document.body.setAttribute("data-p-unselectable-text", "true"), !this.isUnstyled && Lp(document.body, {
        "user-select": "none"
      }), this.$emit("dragstart", t));
    },
    bindGlobalListeners: function() {
      this.draggable && (this.bindDocumentDragListener(), this.bindDocumentDragEndListener()), this.closeOnEscape && this.bindDocumentKeyDownListener();
    },
    unbindGlobalListeners: function() {
      this.unbindDocumentDragListener(), this.unbindDocumentDragEndListener(), this.unbindDocumentKeyDownListener();
    },
    bindDocumentDragListener: function() {
      var t = this;
      this.documentDragListener = function(n) {
        if (t.dragging) {
          var r = Du(t.container), o = Fu(t.container), i = n.pageX - t.lastPageX, s = n.pageY - t.lastPageY, a = t.container.getBoundingClientRect(), l = a.left + i, u = a.top + s, c = Op(), d = getComputedStyle(t.container), f = parseFloat(d.marginLeft), h = parseFloat(d.marginTop);
          t.container.style.position = "fixed", t.keepInViewport ? (l >= t.minX && l + r < c.width && (t.lastPageX = n.pageX, t.container.style.left = l - f + "px"), u >= t.minY && u + o < c.height && (t.lastPageY = n.pageY, t.container.style.top = u - h + "px")) : (t.lastPageX = n.pageX, t.container.style.left = l - f + "px", t.lastPageY = n.pageY, t.container.style.top = u - h + "px");
        }
      }, window.document.addEventListener("mousemove", this.documentDragListener);
    },
    unbindDocumentDragListener: function() {
      this.documentDragListener && (window.document.removeEventListener("mousemove", this.documentDragListener), this.documentDragListener = null);
    },
    bindDocumentDragEndListener: function() {
      var t = this;
      this.documentDragEndListener = function(n) {
        t.dragging && (t.dragging = !1, document.body.removeAttribute("data-p-unselectable-text"), !t.isUnstyled && (document.body.style["user-select"] = ""), t.$emit("dragend", n));
      }, window.document.addEventListener("mouseup", this.documentDragEndListener);
    },
    unbindDocumentDragEndListener: function() {
      this.documentDragEndListener && (window.document.removeEventListener("mouseup", this.documentDragEndListener), this.documentDragEndListener = null);
    }
  },
  computed: {
    maximizeIconComponent: function() {
      return this.maximized ? this.minimizeIcon ? "span" : "WindowMinimizeIcon" : this.maximizeIcon ? "span" : "WindowMaximizeIcon";
    },
    ariaLabelledById: function() {
      return this.header != null || this.$attrs["aria-labelledby"] !== null ? this.$id + "_header" : null;
    },
    closeAriaLabel: function() {
      return this.$primevue.config.locale.aria ? this.$primevue.config.locale.aria.close : void 0;
    },
    dataP: function() {
      return Wn({
        maximized: this.maximized,
        modal: this.modal
      });
    }
  },
  directives: {
    ripple: rc,
    focustrap: yh
  },
  components: {
    Button: oc,
    Portal: ic,
    WindowMinimizeIcon: ec,
    WindowMaximizeIcon: Qu,
    TimesIcon: Zu
  }
};
function Dr(e) {
  "@babel/helpers - typeof";
  return Dr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Dr(e);
}
function $a(e, t) {
  var n = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var r = Object.getOwnPropertySymbols(e);
    t && (r = r.filter(function(o) {
      return Object.getOwnPropertyDescriptor(e, o).enumerable;
    })), n.push.apply(n, r);
  }
  return n;
}
function Na(e) {
  for (var t = 1; t < arguments.length; t++) {
    var n = arguments[t] != null ? arguments[t] : {};
    t % 2 ? $a(Object(n), !0).forEach(function(r) {
      Lh(e, r, n[r]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n)) : $a(Object(n)).forEach(function(r) {
      Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(n, r));
    });
  }
  return e;
}
function Lh(e, t, n) {
  return (t = Ph(t)) in e ? Object.defineProperty(e, t, { value: n, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = n, e;
}
function Ph(e) {
  var t = wh(e, "string");
  return Dr(t) == "symbol" ? t : t + "";
}
function wh(e, t) {
  if (Dr(e) != "object" || !e) return e;
  var n = e[Symbol.toPrimitive];
  if (n !== void 0) {
    var r = n.call(e, t);
    if (Dr(r) != "object") return r;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var kh = ["data-p"], $h = ["aria-labelledby", "aria-modal", "data-p"], Nh = ["id"], Ih = ["data-p"];
function Ah(e, t, n, r, o, i) {
  var s = uo("Button"), a = uo("Portal"), l = au("focustrap");
  return be(), Ct(a, {
    appendTo: e.appendTo
  }, {
    default: Tn(function() {
      return [o.containerVisible ? (be(), Ze("div", de({
        key: 0,
        ref: i.maskRef,
        class: e.cx("mask"),
        style: e.sx("mask", !0, {
          position: e.position,
          modal: e.modal
        }),
        onMousedown: t[1] || (t[1] = function() {
          return i.onMaskMouseDown && i.onMaskMouseDown.apply(i, arguments);
        }),
        onMouseup: t[2] || (t[2] = function() {
          return i.onMaskMouseUp && i.onMaskMouseUp.apply(i, arguments);
        }),
        "data-p": i.dataP
      }, e.ptm("mask")), [ke(jf, de({
        name: "p-dialog",
        onEnter: i.onEnter,
        onAfterEnter: i.onAfterEnter,
        onBeforeLeave: i.onBeforeLeave,
        onLeave: i.onLeave,
        onAfterLeave: i.onAfterLeave,
        appear: ""
      }, e.ptm("transition")), {
        default: Tn(function() {
          return [e.visible ? zl((be(), Ze("div", de({
            key: 0,
            ref: i.containerRef,
            class: e.cx("root"),
            style: e.sx("root"),
            role: "dialog",
            "aria-labelledby": i.ariaLabelledById,
            "aria-modal": e.modal,
            "data-p": i.dataP
          }, e.ptmi("root")), [e.$slots.container ? Xe(e.$slots, "container", {
            key: 0,
            closeCallback: i.close,
            maximizeCallback: function(c) {
              return i.maximize(c);
            }
          }) : (be(), Ze(Ve, {
            key: 1
          }, [e.showHeader ? (be(), Ze("div", de({
            key: 0,
            ref: i.headerContainerRef,
            class: e.cx("header"),
            onMousedown: t[0] || (t[0] = function() {
              return i.initDrag && i.initDrag.apply(i, arguments);
            })
          }, e.ptm("header")), [Xe(e.$slots, "header", {
            class: Bn(e.cx("title"))
          }, function() {
            return [e.header ? (be(), Ze("span", de({
              key: 0,
              id: i.ariaLabelledById,
              class: e.cx("title")
            }, e.ptm("title")), mr(e.header), 17, Nh)) : vt("", !0)];
          }), Ft("div", de({
            class: e.cx("headerActions")
          }, e.ptm("headerActions")), [e.maximizable ? Xe(e.$slots, "maximizebutton", {
            key: 0,
            maximized: o.maximized,
            maximizeCallback: function(c) {
              return i.maximize(c);
            }
          }, function() {
            return [ke(s, de({
              ref: i.maximizableRef,
              autofocus: o.focusableMax,
              class: e.cx("pcMaximizeButton"),
              onClick: i.maximize,
              tabindex: e.maximizable ? "0" : "-1",
              unstyled: e.unstyled
            }, e.maximizeButtonProps, {
              pt: e.ptm("pcMaximizeButton"),
              "data-pc-group-section": "headericon"
            }), {
              icon: Tn(function(u) {
                return [Xe(e.$slots, "maximizeicon", {
                  maximized: o.maximized
                }, function() {
                  return [(be(), Ct(ui(i.maximizeIconComponent), de({
                    class: [u.class, o.maximized ? e.minimizeIcon : e.maximizeIcon]
                  }, e.ptm("pcMaximizeButton").icon), null, 16, ["class"]))];
                })];
              }),
              _: 3
            }, 16, ["autofocus", "class", "onClick", "tabindex", "unstyled", "pt"])];
          }) : vt("", !0), e.closable ? Xe(e.$slots, "closebutton", {
            key: 1,
            closeCallback: i.close
          }, function() {
            return [ke(s, de({
              ref: i.closeButtonRef,
              autofocus: o.focusableClose,
              class: e.cx("pcCloseButton"),
              onClick: i.close,
              "aria-label": i.closeAriaLabel,
              unstyled: e.unstyled
            }, e.closeButtonProps, {
              pt: e.ptm("pcCloseButton"),
              "data-pc-group-section": "headericon"
            }), {
              icon: Tn(function(u) {
                return [Xe(e.$slots, "closeicon", {}, function() {
                  return [(be(), Ct(ui(e.closeIcon ? "span" : "TimesIcon"), de({
                    class: [e.closeIcon, u.class]
                  }, e.ptm("pcCloseButton").icon), null, 16, ["class"]))];
                })];
              }),
              _: 3
            }, 16, ["autofocus", "class", "onClick", "aria-label", "unstyled", "pt"])];
          }) : vt("", !0)], 16)], 16)) : vt("", !0), Ft("div", de({
            ref: i.contentRef,
            class: [e.cx("content"), e.contentClass],
            style: e.contentStyle,
            "data-p": i.dataP
          }, Na(Na({}, e.contentProps), e.ptm("content"))), [Xe(e.$slots, "default")], 16, Ih), e.footer || e.$slots.footer ? (be(), Ze("div", de({
            key: 1,
            ref: i.footerContainerRef,
            class: e.cx("footer")
          }, e.ptm("footer")), [Xe(e.$slots, "footer", {}, function() {
            return [qi(mr(e.footer), 1)];
          })], 16)) : vt("", !0)], 64))], 16, $h)), [[l, {
            disabled: !e.modal
          }]]) : vt("", !0)];
        }),
        _: 3
      }, 16, ["onEnter", "onAfterEnter", "onBeforeLeave", "onLeave", "onAfterLeave"])], 16, kh)) : vt("", !0)];
    }),
    _: 3
  }, 8, ["appendTo"]);
}
sc.render = Ah;
const xh = /* @__PURE__ */ Ur({
  __name: "App",
  setup(e) {
    const t = De(), n = De(), r = De(!1), o = De(null), i = () => {
      var s;
      if ((s = t.value) != null && s.parentElement) {
        const a = t.value.parentElement;
        r.value ? a.classList.remove("h-full") : a.classList.add("h-full");
      }
    };
    return ft(r, () => {
      i();
    }), Vr(async () => {
      t.value && (i(), o.value = new MutationObserver((s) => {
        s.forEach((a) => {
          a.type === "attributes" && a.attributeName === "maximized" && (r.value = a.target.getAttribute("maximized") === "true");
        });
      }), o.value.observe(t.value, {
        attributes: !0,
        attributeFilter: ["maximized"]
      }));
    }), Bi(() => {
      var s;
      (s = t.value) != null && s.parentElement && t.value.parentElement.classList.remove("h-full"), o.value && (o.value.disconnect(), o.value = null);
    }), (s, a) => (be(), Ze("div", {
      ref_key: "viewerContentRef",
      ref: t,
      class: "flex w-full h-full"
    }, [
      Ft("div", {
        ref_key: "mainContentRef",
        ref: n,
        class: "flex-1 relative"
      }, a[0] || (a[0] = [
        Ft("iframe", {
          src: "/opencut",
          class: "demo-iframe h-full w-full"
        }, null, -1)
      ]), 512)
    ], 512));
  }
}), Dh = (e, t) => {
  const n = e.__vccOpts || e;
  for (const [r, o] of t)
    n[r] = o;
  return n;
}, Rh = /* @__PURE__ */ Dh(xh, [["__scopeId", "data-v-5772e9c3"]]), Mh = /* @__PURE__ */ Ur({
  __name: "Root",
  setup(e, { expose: t }) {
    const n = De(!1), r = De(null);
    return t({ open: () => {
      n.value = !0;
    }, close: () => {
      n.value = !1;
    } }), (s, a) => (be(), Ct(jl(sc), {
      visible: n.value,
      "onUpdate:visible": a[0] || (a[0] = (l) => n.value = l),
      header: "ComfyUI OpenCut",
      style: { width: "80vw", height: "80vh" },
      maximizable: !0,
      modal: !0,
      closable: !0,
      draggable: !1,
      "content-class": "h-full"
    }, {
      default: Tn(() => [
        ke(Rh, {
          ref_key: "appRef",
          ref: r
        }, null, 512)
      ]),
      _: 1
    }, 8, ["visible"]));
  }
});
/*!
  * shared v9.14.5
  * (c) 2025 kazuya kawaguchi
  * Released under the MIT License.
  */
function Fh(e, t) {
  typeof console < "u" && (console.warn("[intlify] " + e), t && console.warn(t.stack));
}
const go = typeof window < "u", dn = (e, t = !1) => t ? Symbol.for(e) : Symbol(e), jh = (e, t, n) => Uh({ l: e, k: t, s: n }), Uh = (e) => JSON.stringify(e).replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029").replace(/\u0027/g, "\\u0027"), Le = (e) => typeof e == "number" && isFinite(e), Vh = (e) => lc(e) === "[object Date]", cn = (e) => lc(e) === "[object RegExp]", No = (e) => J(e) && Object.keys(e).length === 0, Fe = Object.assign, Hh = Object.create, pe = (e = null) => Hh(e);
let Ia;
const Gt = () => Ia || (Ia = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : pe());
function Aa(e) {
  return e.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;").replace(/\//g, "&#x2F;").replace(/=/g, "&#x3D;");
}
function xa(e) {
  return e.replace(/&(?![a-zA-Z0-9#]{2,6};)/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&apos;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function Wh(e) {
  return e = e.replace(/(\w+)\s*=\s*"([^"]*)"/g, (r, o, i) => `${o}="${xa(i)}"`), e = e.replace(/(\w+)\s*=\s*'([^']*)'/g, (r, o, i) => `${o}='${xa(i)}'`), /\s*on\w+\s*=\s*["']?[^"'>]+["']?/gi.test(e) && (e = e.replace(/(\s+)(on)(\w+\s*=)/gi, "$1&#111;n$3")), [
    // In href, src, action, formaction attributes
    /(\s+(?:href|src|action|formaction)\s*=\s*["']?)\s*javascript:/gi,
    // In style attributes within url()
    /(style\s*=\s*["'][^"']*url\s*\(\s*)javascript:/gi
  ].forEach((r) => {
    e = e.replace(r, "$1javascript&#58;");
  }), e;
}
const Bh = Object.prototype.hasOwnProperty;
function _t(e, t) {
  return Bh.call(e, t);
}
const Se = Array.isArray, ve = (e) => typeof e == "function", W = (e) => typeof e == "string", ee = (e) => typeof e == "boolean", se = (e) => e !== null && typeof e == "object", Kh = (e) => se(e) && ve(e.then) && ve(e.catch), ac = Object.prototype.toString, lc = (e) => ac.call(e), J = (e) => {
  if (!se(e))
    return !1;
  const t = Object.getPrototypeOf(e);
  return t === null || t.constructor === Object;
}, Yh = (e) => e == null ? "" : Se(e) || J(e) && e.toString === ac ? JSON.stringify(e, null, 2) : String(e);
function zh(e, t = "") {
  return e.reduce((n, r, o) => o === 0 ? n + r : n + t + r, "");
}
function Io(e) {
  let t = e;
  return () => ++t;
}
const Qr = (e) => !se(e) || Se(e);
function io(e, t) {
  if (Qr(e) || Qr(t))
    throw new Error("Invalid value");
  const n = [{ src: e, des: t }];
  for (; n.length; ) {
    const { src: r, des: o } = n.pop();
    Object.keys(r).forEach((i) => {
      i !== "__proto__" && (se(r[i]) && !se(o[i]) && (o[i] = Array.isArray(r[i]) ? [] : pe()), Qr(o[i]) || Qr(r[i]) ? o[i] = r[i] : n.push({ src: r[i], des: o[i] }));
    });
  }
}
/*!
  * message-compiler v9.14.5
  * (c) 2025 kazuya kawaguchi
  * Released under the MIT License.
  */
function Gh(e, t, n) {
  return { line: e, column: t, offset: n };
}
function bo(e, t, n) {
  return { start: e, end: t };
}
const Xh = /\{([0-9a-zA-Z]+)\}/g;
function uc(e, ...t) {
  return t.length === 1 && Jh(t[0]) && (t = t[0]), (!t || !t.hasOwnProperty) && (t = {}), e.replace(Xh, (n, r) => t.hasOwnProperty(r) ? t[r] : "");
}
const cc = Object.assign, Da = (e) => typeof e == "string", Jh = (e) => e !== null && typeof e == "object";
function dc(e, t = "") {
  return e.reduce((n, r, o) => o === 0 ? n + r : n + t + r, "");
}
const ns = {
  USE_MODULO_SYNTAX: 1,
  __EXTEND_POINT__: 2
}, qh = {
  [ns.USE_MODULO_SYNTAX]: "Use modulo before '{{0}}'."
};
function Zh(e, t, ...n) {
  const r = uc(qh[e], ...n || []), o = { message: String(r), code: e };
  return t && (o.location = t), o;
}
const G = {
  // tokenizer error codes
  EXPECTED_TOKEN: 1,
  INVALID_TOKEN_IN_PLACEHOLDER: 2,
  UNTERMINATED_SINGLE_QUOTE_IN_PLACEHOLDER: 3,
  UNKNOWN_ESCAPE_SEQUENCE: 4,
  INVALID_UNICODE_ESCAPE_SEQUENCE: 5,
  UNBALANCED_CLOSING_BRACE: 6,
  UNTERMINATED_CLOSING_BRACE: 7,
  EMPTY_PLACEHOLDER: 8,
  NOT_ALLOW_NEST_PLACEHOLDER: 9,
  INVALID_LINKED_FORMAT: 10,
  // parser error codes
  MUST_HAVE_MESSAGES_IN_PLURAL: 11,
  UNEXPECTED_EMPTY_LINKED_MODIFIER: 12,
  UNEXPECTED_EMPTY_LINKED_KEY: 13,
  UNEXPECTED_LEXICAL_ANALYSIS: 14,
  // generator error codes
  UNHANDLED_CODEGEN_NODE_TYPE: 15,
  // minifier error codes
  UNHANDLED_MINIFIER_NODE_TYPE: 16,
  // Special value for higher-order compilers to pick up the last code
  // to avoid collision of error codes. This should always be kept as the last
  // item.
  __EXTEND_POINT__: 17
}, Qh = {
  // tokenizer error messages
  [G.EXPECTED_TOKEN]: "Expected token: '{0}'",
  [G.INVALID_TOKEN_IN_PLACEHOLDER]: "Invalid token in placeholder: '{0}'",
  [G.UNTERMINATED_SINGLE_QUOTE_IN_PLACEHOLDER]: "Unterminated single quote in placeholder",
  [G.UNKNOWN_ESCAPE_SEQUENCE]: "Unknown escape sequence: \\{0}",
  [G.INVALID_UNICODE_ESCAPE_SEQUENCE]: "Invalid unicode escape sequence: {0}",
  [G.UNBALANCED_CLOSING_BRACE]: "Unbalanced closing brace",
  [G.UNTERMINATED_CLOSING_BRACE]: "Unterminated closing brace",
  [G.EMPTY_PLACEHOLDER]: "Empty placeholder",
  [G.NOT_ALLOW_NEST_PLACEHOLDER]: "Not allowed nest placeholder",
  [G.INVALID_LINKED_FORMAT]: "Invalid linked format",
  // parser error messages
  [G.MUST_HAVE_MESSAGES_IN_PLURAL]: "Plural must have messages",
  [G.UNEXPECTED_EMPTY_LINKED_MODIFIER]: "Unexpected empty linked modifier",
  [G.UNEXPECTED_EMPTY_LINKED_KEY]: "Unexpected empty linked key",
  [G.UNEXPECTED_LEXICAL_ANALYSIS]: "Unexpected lexical analysis in token: '{0}'",
  // generator error messages
  [G.UNHANDLED_CODEGEN_NODE_TYPE]: "unhandled codegen node type: '{0}'",
  // minimizer error messages
  [G.UNHANDLED_MINIFIER_NODE_TYPE]: "unhandled mimifier node type: '{0}'"
};
function zn(e, t, n = {}) {
  const { domain: r, messages: o, args: i } = n, s = uc((o || Qh)[e] || "", ...i || []), a = new SyntaxError(String(s));
  return a.code = e, t && (a.location = t), a.domain = r, a;
}
function eg(e) {
  throw e;
}
const Wt = " ", tg = "\r", ze = `
`, ng = "\u2028", rg = "\u2029";
function og(e) {
  const t = e;
  let n = 0, r = 1, o = 1, i = 0;
  const s = (A) => t[A] === tg && t[A + 1] === ze, a = (A) => t[A] === ze, l = (A) => t[A] === rg, u = (A) => t[A] === ng, c = (A) => s(A) || a(A) || l(A) || u(A), d = () => n, f = () => r, h = () => o, _ = () => i, E = (A) => s(A) || l(A) || u(A) ? ze : t[A], w = () => E(n), P = () => E(n + i);
  function M() {
    return i = 0, c(n) && (r++, o = 0), s(n) && n++, n++, o++, t[n];
  }
  function S() {
    return s(n + i) && i++, i++, t[n + i];
  }
  function g() {
    n = 0, r = 1, o = 1, i = 0;
  }
  function O(A = 0) {
    i = A;
  }
  function L() {
    const A = n + i;
    for (; A !== n; )
      M();
    i = 0;
  }
  return {
    index: d,
    line: f,
    column: h,
    peekOffset: _,
    charAt: E,
    currentChar: w,
    currentPeek: P,
    next: M,
    peek: S,
    reset: g,
    resetPeek: O,
    skipToPeek: L
  };
}
const tn = void 0, ig = ".", Ra = "'", sg = "tokenizer";
function ag(e, t = {}) {
  const n = t.location !== !1, r = og(e), o = () => r.index(), i = () => Gh(r.line(), r.column(), r.index()), s = i(), a = o(), l = {
    currentType: 14,
    offset: a,
    startLoc: s,
    endLoc: s,
    lastType: 14,
    lastOffset: a,
    lastStartLoc: s,
    lastEndLoc: s,
    braceNest: 0,
    inLinked: !1,
    text: ""
  }, u = () => l, { onError: c } = t;
  function d(p, m, T, ...k) {
    const H = u();
    if (m.column += T, m.offset += T, c) {
      const j = n ? bo(H.startLoc, m) : null, C = zn(p, j, {
        domain: sg,
        args: k
      });
      c(C);
    }
  }
  function f(p, m, T) {
    p.endLoc = i(), p.currentType = m;
    const k = { type: m };
    return n && (k.loc = bo(p.startLoc, p.endLoc)), T != null && (k.value = T), k;
  }
  const h = (p) => f(
    p,
    14
    /* TokenTypes.EOF */
  );
  function _(p, m) {
    return p.currentChar() === m ? (p.next(), m) : (d(G.EXPECTED_TOKEN, i(), 0, m), "");
  }
  function E(p) {
    let m = "";
    for (; p.currentPeek() === Wt || p.currentPeek() === ze; )
      m += p.currentPeek(), p.peek();
    return m;
  }
  function w(p) {
    const m = E(p);
    return p.skipToPeek(), m;
  }
  function P(p) {
    if (p === tn)
      return !1;
    const m = p.charCodeAt(0);
    return m >= 97 && m <= 122 || // a-z
    m >= 65 && m <= 90 || // A-Z
    m === 95;
  }
  function M(p) {
    if (p === tn)
      return !1;
    const m = p.charCodeAt(0);
    return m >= 48 && m <= 57;
  }
  function S(p, m) {
    const { currentType: T } = m;
    if (T !== 2)
      return !1;
    E(p);
    const k = P(p.currentPeek());
    return p.resetPeek(), k;
  }
  function g(p, m) {
    const { currentType: T } = m;
    if (T !== 2)
      return !1;
    E(p);
    const k = p.currentPeek() === "-" ? p.peek() : p.currentPeek(), H = M(k);
    return p.resetPeek(), H;
  }
  function O(p, m) {
    const { currentType: T } = m;
    if (T !== 2)
      return !1;
    E(p);
    const k = p.currentPeek() === Ra;
    return p.resetPeek(), k;
  }
  function L(p, m) {
    const { currentType: T } = m;
    if (T !== 8)
      return !1;
    E(p);
    const k = p.currentPeek() === ".";
    return p.resetPeek(), k;
  }
  function A(p, m) {
    const { currentType: T } = m;
    if (T !== 9)
      return !1;
    E(p);
    const k = P(p.currentPeek());
    return p.resetPeek(), k;
  }
  function F(p, m) {
    const { currentType: T } = m;
    if (!(T === 8 || T === 12))
      return !1;
    E(p);
    const k = p.currentPeek() === ":";
    return p.resetPeek(), k;
  }
  function $(p, m) {
    const { currentType: T } = m;
    if (T !== 10)
      return !1;
    const k = () => {
      const j = p.currentPeek();
      return j === "{" ? P(p.peek()) : j === "@" || j === "%" || j === "|" || j === ":" || j === "." || j === Wt || !j ? !1 : j === ze ? (p.peek(), k()) : R(p, !1);
    }, H = k();
    return p.resetPeek(), H;
  }
  function B(p) {
    E(p);
    const m = p.currentPeek() === "|";
    return p.resetPeek(), m;
  }
  function Y(p) {
    const m = E(p), T = p.currentPeek() === "%" && p.peek() === "{";
    return p.resetPeek(), {
      isModulo: T,
      hasSpace: m.length > 0
    };
  }
  function R(p, m = !0) {
    const T = (H = !1, j = "", C = !1) => {
      const I = p.currentPeek();
      return I === "{" ? j === "%" ? !1 : H : I === "@" || !I ? j === "%" ? !0 : H : I === "%" ? (p.peek(), T(H, "%", !0)) : I === "|" ? j === "%" || C ? !0 : !(j === Wt || j === ze) : I === Wt ? (p.peek(), T(!0, Wt, C)) : I === ze ? (p.peek(), T(!0, ze, C)) : !0;
    }, k = T();
    return m && p.resetPeek(), k;
  }
  function z(p, m) {
    const T = p.currentChar();
    return T === tn ? tn : m(T) ? (p.next(), T) : null;
  }
  function ae(p) {
    const m = p.charCodeAt(0);
    return m >= 97 && m <= 122 || // a-z
    m >= 65 && m <= 90 || // A-Z
    m >= 48 && m <= 57 || // 0-9
    m === 95 || // _
    m === 36;
  }
  function Te(p) {
    return z(p, ae);
  }
  function ne(p) {
    const m = p.charCodeAt(0);
    return m >= 97 && m <= 122 || // a-z
    m >= 65 && m <= 90 || // A-Z
    m >= 48 && m <= 57 || // 0-9
    m === 95 || // _
    m === 36 || // $
    m === 45;
  }
  function te(p) {
    return z(p, ne);
  }
  function Q(p) {
    const m = p.charCodeAt(0);
    return m >= 48 && m <= 57;
  }
  function Pe(p) {
    return z(p, Q);
  }
  function we(p) {
    const m = p.charCodeAt(0);
    return m >= 48 && m <= 57 || // 0-9
    m >= 65 && m <= 70 || // A-F
    m >= 97 && m <= 102;
  }
  function ue(p) {
    return z(p, we);
  }
  function he(p) {
    let m = "", T = "";
    for (; m = Pe(p); )
      T += m;
    return T;
  }
  function it(p) {
    w(p);
    const m = p.currentChar();
    return m !== "%" && d(G.EXPECTED_TOKEN, i(), 0, m), p.next(), "%";
  }
  function Be(p) {
    let m = "";
    for (; ; ) {
      const T = p.currentChar();
      if (T === "{" || T === "}" || T === "@" || T === "|" || !T)
        break;
      if (T === "%")
        if (R(p))
          m += T, p.next();
        else
          break;
      else if (T === Wt || T === ze)
        if (R(p))
          m += T, p.next();
        else {
          if (B(p))
            break;
          m += T, p.next();
        }
      else
        m += T, p.next();
    }
    return m;
  }
  function Lt(p) {
    w(p);
    let m = "", T = "";
    for (; m = te(p); )
      T += m;
    return p.currentChar() === tn && d(G.UNTERMINATED_CLOSING_BRACE, i(), 0), T;
  }
  function Qe(p) {
    w(p);
    let m = "";
    return p.currentChar() === "-" ? (p.next(), m += `-${he(p)}`) : m += he(p), p.currentChar() === tn && d(G.UNTERMINATED_CLOSING_BRACE, i(), 0), m;
  }
  function mt(p) {
    return p !== Ra && p !== ze;
  }
  function ht(p) {
    w(p), _(p, "'");
    let m = "", T = "";
    for (; m = z(p, mt); )
      m === "\\" ? T += Pt(p) : T += m;
    const k = p.currentChar();
    return k === ze || k === tn ? (d(G.UNTERMINATED_SINGLE_QUOTE_IN_PLACEHOLDER, i(), 0), k === ze && (p.next(), _(p, "'")), T) : (_(p, "'"), T);
  }
  function Pt(p) {
    const m = p.currentChar();
    switch (m) {
      case "\\":
      case "'":
        return p.next(), `\\${m}`;
      case "u":
        return et(p, m, 4);
      case "U":
        return et(p, m, 6);
      default:
        return d(G.UNKNOWN_ESCAPE_SEQUENCE, i(), 0, m), "";
    }
  }
  function et(p, m, T) {
    _(p, m);
    let k = "";
    for (let H = 0; H < T; H++) {
      const j = ue(p);
      if (!j) {
        d(G.INVALID_UNICODE_ESCAPE_SEQUENCE, i(), 0, `\\${m}${k}${p.currentChar()}`);
        break;
      }
      k += j;
    }
    return `\\${m}${k}`;
  }
  function Qt(p) {
    return p !== "{" && p !== "}" && p !== Wt && p !== ze;
  }
  function b(p) {
    w(p);
    let m = "", T = "";
    for (; m = z(p, Qt); )
      T += m;
    return T;
  }
  function y(p) {
    let m = "", T = "";
    for (; m = Te(p); )
      T += m;
    return T;
  }
  function v(p) {
    const m = (T) => {
      const k = p.currentChar();
      return k === "{" || k === "%" || k === "@" || k === "|" || k === "(" || k === ")" || !k || k === Wt ? T : (T += k, p.next(), m(T));
    };
    return m("");
  }
  function N(p) {
    w(p);
    const m = _(
      p,
      "|"
      /* TokenChars.Pipe */
    );
    return w(p), m;
  }
  function x(p, m) {
    let T = null;
    switch (p.currentChar()) {
      case "{":
        return m.braceNest >= 1 && d(G.NOT_ALLOW_NEST_PLACEHOLDER, i(), 0), p.next(), T = f(
          m,
          2,
          "{"
          /* TokenChars.BraceLeft */
        ), w(p), m.braceNest++, T;
      case "}":
        return m.braceNest > 0 && m.currentType === 2 && d(G.EMPTY_PLACEHOLDER, i(), 0), p.next(), T = f(
          m,
          3,
          "}"
          /* TokenChars.BraceRight */
        ), m.braceNest--, m.braceNest > 0 && w(p), m.inLinked && m.braceNest === 0 && (m.inLinked = !1), T;
      case "@":
        return m.braceNest > 0 && d(G.UNTERMINATED_CLOSING_BRACE, i(), 0), T = D(p, m) || h(m), m.braceNest = 0, T;
      default: {
        let H = !0, j = !0, C = !0;
        if (B(p))
          return m.braceNest > 0 && d(G.UNTERMINATED_CLOSING_BRACE, i(), 0), T = f(m, 1, N(p)), m.braceNest = 0, m.inLinked = !1, T;
        if (m.braceNest > 0 && (m.currentType === 5 || m.currentType === 6 || m.currentType === 7))
          return d(G.UNTERMINATED_CLOSING_BRACE, i(), 0), m.braceNest = 0, V(p, m);
        if (H = S(p, m))
          return T = f(m, 5, Lt(p)), w(p), T;
        if (j = g(p, m))
          return T = f(m, 6, Qe(p)), w(p), T;
        if (C = O(p, m))
          return T = f(m, 7, ht(p)), w(p), T;
        if (!H && !j && !C)
          return T = f(m, 13, b(p)), d(G.INVALID_TOKEN_IN_PLACEHOLDER, i(), 0, T.value), w(p), T;
        break;
      }
    }
    return T;
  }
  function D(p, m) {
    const { currentType: T } = m;
    let k = null;
    const H = p.currentChar();
    switch ((T === 8 || T === 9 || T === 12 || T === 10) && (H === ze || H === Wt) && d(G.INVALID_LINKED_FORMAT, i(), 0), H) {
      case "@":
        return p.next(), k = f(
          m,
          8,
          "@"
          /* TokenChars.LinkedAlias */
        ), m.inLinked = !0, k;
      case ".":
        return w(p), p.next(), f(
          m,
          9,
          "."
          /* TokenChars.LinkedDot */
        );
      case ":":
        return w(p), p.next(), f(
          m,
          10,
          ":"
          /* TokenChars.LinkedDelimiter */
        );
      default:
        return B(p) ? (k = f(m, 1, N(p)), m.braceNest = 0, m.inLinked = !1, k) : L(p, m) || F(p, m) ? (w(p), D(p, m)) : A(p, m) ? (w(p), f(m, 12, y(p))) : $(p, m) ? (w(p), H === "{" ? x(p, m) || k : f(m, 11, v(p))) : (T === 8 && d(G.INVALID_LINKED_FORMAT, i(), 0), m.braceNest = 0, m.inLinked = !1, V(p, m));
    }
  }
  function V(p, m) {
    let T = {
      type: 14
      /* TokenTypes.EOF */
    };
    if (m.braceNest > 0)
      return x(p, m) || h(m);
    if (m.inLinked)
      return D(p, m) || h(m);
    switch (p.currentChar()) {
      case "{":
        return x(p, m) || h(m);
      case "}":
        return d(G.UNBALANCED_CLOSING_BRACE, i(), 0), p.next(), f(
          m,
          3,
          "}"
          /* TokenChars.BraceRight */
        );
      case "@":
        return D(p, m) || h(m);
      default: {
        if (B(p))
          return T = f(m, 1, N(p)), m.braceNest = 0, m.inLinked = !1, T;
        const { isModulo: H, hasSpace: j } = Y(p);
        if (H)
          return j ? f(m, 0, Be(p)) : f(m, 4, it(p));
        if (R(p))
          return f(m, 0, Be(p));
        break;
      }
    }
    return T;
  }
  function U() {
    const { currentType: p, offset: m, startLoc: T, endLoc: k } = l;
    return l.lastType = p, l.lastOffset = m, l.lastStartLoc = T, l.lastEndLoc = k, l.offset = o(), l.startLoc = i(), r.currentChar() === tn ? f(
      l,
      14
      /* TokenTypes.EOF */
    ) : V(r, l);
  }
  return {
    nextToken: U,
    currentOffset: o,
    currentPosition: i,
    context: u
  };
}
const lg = "parser", ug = /(?:\\\\|\\'|\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{6}))/g;
function cg(e, t, n) {
  switch (e) {
    case "\\\\":
      return "\\";
    // eslint-disable-next-line no-useless-escape
    case "\\'":
      return "'";
    default: {
      const r = parseInt(t || n, 16);
      return r <= 55295 || r >= 57344 ? String.fromCodePoint(r) : "�";
    }
  }
}
function dg(e = {}) {
  const t = e.location !== !1, { onError: n, onWarn: r } = e;
  function o(S, g, O, L, ...A) {
    const F = S.currentPosition();
    if (F.offset += L, F.column += L, n) {
      const $ = t ? bo(O, F) : null, B = zn(g, $, {
        domain: lg,
        args: A
      });
      n(B);
    }
  }
  function i(S, g, O, L, ...A) {
    const F = S.currentPosition();
    if (F.offset += L, F.column += L, r) {
      const $ = t ? bo(O, F) : null;
      r(Zh(g, $, A));
    }
  }
  function s(S, g, O) {
    const L = { type: S };
    return t && (L.start = g, L.end = g, L.loc = { start: O, end: O }), L;
  }
  function a(S, g, O, L) {
    t && (S.end = g, S.loc && (S.loc.end = O));
  }
  function l(S, g) {
    const O = S.context(), L = s(3, O.offset, O.startLoc);
    return L.value = g, a(L, S.currentOffset(), S.currentPosition()), L;
  }
  function u(S, g) {
    const O = S.context(), { lastOffset: L, lastStartLoc: A } = O, F = s(5, L, A);
    return F.index = parseInt(g, 10), S.nextToken(), a(F, S.currentOffset(), S.currentPosition()), F;
  }
  function c(S, g, O) {
    const L = S.context(), { lastOffset: A, lastStartLoc: F } = L, $ = s(4, A, F);
    return $.key = g, O === !0 && ($.modulo = !0), S.nextToken(), a($, S.currentOffset(), S.currentPosition()), $;
  }
  function d(S, g) {
    const O = S.context(), { lastOffset: L, lastStartLoc: A } = O, F = s(9, L, A);
    return F.value = g.replace(ug, cg), S.nextToken(), a(F, S.currentOffset(), S.currentPosition()), F;
  }
  function f(S) {
    const g = S.nextToken(), O = S.context(), { lastOffset: L, lastStartLoc: A } = O, F = s(8, L, A);
    return g.type !== 12 ? (o(S, G.UNEXPECTED_EMPTY_LINKED_MODIFIER, O.lastStartLoc, 0), F.value = "", a(F, L, A), {
      nextConsumeToken: g,
      node: F
    }) : (g.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, O.lastStartLoc, 0, bt(g)), F.value = g.value || "", a(F, S.currentOffset(), S.currentPosition()), {
      node: F
    });
  }
  function h(S, g) {
    const O = S.context(), L = s(7, O.offset, O.startLoc);
    return L.value = g, a(L, S.currentOffset(), S.currentPosition()), L;
  }
  function _(S) {
    const g = S.context(), O = s(6, g.offset, g.startLoc);
    let L = S.nextToken();
    if (L.type === 9) {
      const A = f(S);
      O.modifier = A.node, L = A.nextConsumeToken || S.nextToken();
    }
    switch (L.type !== 10 && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(L)), L = S.nextToken(), L.type === 2 && (L = S.nextToken()), L.type) {
      case 11:
        L.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(L)), O.key = h(S, L.value || "");
        break;
      case 5:
        L.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(L)), O.key = c(S, L.value || "");
        break;
      case 6:
        L.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(L)), O.key = u(S, L.value || "");
        break;
      case 7:
        L.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(L)), O.key = d(S, L.value || "");
        break;
      default: {
        o(S, G.UNEXPECTED_EMPTY_LINKED_KEY, g.lastStartLoc, 0);
        const A = S.context(), F = s(7, A.offset, A.startLoc);
        return F.value = "", a(F, A.offset, A.startLoc), O.key = F, a(O, A.offset, A.startLoc), {
          nextConsumeToken: L,
          node: O
        };
      }
    }
    return a(O, S.currentOffset(), S.currentPosition()), {
      node: O
    };
  }
  function E(S) {
    const g = S.context(), O = g.currentType === 1 ? S.currentOffset() : g.offset, L = g.currentType === 1 ? g.endLoc : g.startLoc, A = s(2, O, L);
    A.items = [];
    let F = null, $ = null;
    do {
      const R = F || S.nextToken();
      switch (F = null, R.type) {
        case 0:
          R.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(R)), A.items.push(l(S, R.value || ""));
          break;
        case 6:
          R.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(R)), A.items.push(u(S, R.value || ""));
          break;
        case 4:
          $ = !0;
          break;
        case 5:
          R.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(R)), A.items.push(c(S, R.value || "", !!$)), $ && (i(S, ns.USE_MODULO_SYNTAX, g.lastStartLoc, 0, bt(R)), $ = null);
          break;
        case 7:
          R.value == null && o(S, G.UNEXPECTED_LEXICAL_ANALYSIS, g.lastStartLoc, 0, bt(R)), A.items.push(d(S, R.value || ""));
          break;
        case 8: {
          const z = _(S);
          A.items.push(z.node), F = z.nextConsumeToken || null;
          break;
        }
      }
    } while (g.currentType !== 14 && g.currentType !== 1);
    const B = g.currentType === 1 ? g.lastOffset : S.currentOffset(), Y = g.currentType === 1 ? g.lastEndLoc : S.currentPosition();
    return a(A, B, Y), A;
  }
  function w(S, g, O, L) {
    const A = S.context();
    let F = L.items.length === 0;
    const $ = s(1, g, O);
    $.cases = [], $.cases.push(L);
    do {
      const B = E(S);
      F || (F = B.items.length === 0), $.cases.push(B);
    } while (A.currentType !== 14);
    return F && o(S, G.MUST_HAVE_MESSAGES_IN_PLURAL, O, 0), a($, S.currentOffset(), S.currentPosition()), $;
  }
  function P(S) {
    const g = S.context(), { offset: O, startLoc: L } = g, A = E(S);
    return g.currentType === 14 ? A : w(S, O, L, A);
  }
  function M(S) {
    const g = ag(S, cc({}, e)), O = g.context(), L = s(0, O.offset, O.startLoc);
    return t && L.loc && (L.loc.source = S), L.body = P(g), e.onCacheKey && (L.cacheKey = e.onCacheKey(S)), O.currentType !== 14 && o(g, G.UNEXPECTED_LEXICAL_ANALYSIS, O.lastStartLoc, 0, S[O.offset] || ""), a(L, g.currentOffset(), g.currentPosition()), L;
  }
  return { parse: M };
}
function bt(e) {
  if (e.type === 14)
    return "EOF";
  const t = (e.value || "").replace(/\r?\n/gu, "\\n");
  return t.length > 10 ? t.slice(0, 9) + "…" : t;
}
function fg(e, t = {}) {
  const n = {
    ast: e,
    helpers: /* @__PURE__ */ new Set()
  };
  return { context: () => n, helper: (i) => (n.helpers.add(i), i) };
}
function Ma(e, t) {
  for (let n = 0; n < e.length; n++)
    rs(e[n], t);
}
function rs(e, t) {
  switch (e.type) {
    case 1:
      Ma(e.cases, t), t.helper(
        "plural"
        /* HelperNameMap.PLURAL */
      );
      break;
    case 2:
      Ma(e.items, t);
      break;
    case 6: {
      rs(e.key, t), t.helper(
        "linked"
        /* HelperNameMap.LINKED */
      ), t.helper(
        "type"
        /* HelperNameMap.TYPE */
      );
      break;
    }
    case 5:
      t.helper(
        "interpolate"
        /* HelperNameMap.INTERPOLATE */
      ), t.helper(
        "list"
        /* HelperNameMap.LIST */
      );
      break;
    case 4:
      t.helper(
        "interpolate"
        /* HelperNameMap.INTERPOLATE */
      ), t.helper(
        "named"
        /* HelperNameMap.NAMED */
      );
      break;
  }
}
function pg(e, t = {}) {
  const n = fg(e);
  n.helper(
    "normalize"
    /* HelperNameMap.NORMALIZE */
  ), e.body && rs(e.body, n);
  const r = n.context();
  e.helpers = Array.from(r.helpers);
}
function mg(e) {
  const t = e.body;
  return t.type === 2 ? Fa(t) : t.cases.forEach((n) => Fa(n)), e;
}
function Fa(e) {
  if (e.items.length === 1) {
    const t = e.items[0];
    (t.type === 3 || t.type === 9) && (e.static = t.value, delete t.value);
  } else {
    const t = [];
    for (let n = 0; n < e.items.length; n++) {
      const r = e.items[n];
      if (!(r.type === 3 || r.type === 9) || r.value == null)
        break;
      t.push(r.value);
    }
    if (t.length === e.items.length) {
      e.static = dc(t);
      for (let n = 0; n < e.items.length; n++) {
        const r = e.items[n];
        (r.type === 3 || r.type === 9) && delete r.value;
      }
    }
  }
}
const hg = "minifier";
function xn(e) {
  switch (e.t = e.type, e.type) {
    case 0: {
      const t = e;
      xn(t.body), t.b = t.body, delete t.body;
      break;
    }
    case 1: {
      const t = e, n = t.cases;
      for (let r = 0; r < n.length; r++)
        xn(n[r]);
      t.c = n, delete t.cases;
      break;
    }
    case 2: {
      const t = e, n = t.items;
      for (let r = 0; r < n.length; r++)
        xn(n[r]);
      t.i = n, delete t.items, t.static && (t.s = t.static, delete t.static);
      break;
    }
    case 3:
    case 9:
    case 8:
    case 7: {
      const t = e;
      t.value && (t.v = t.value, delete t.value);
      break;
    }
    case 6: {
      const t = e;
      xn(t.key), t.k = t.key, delete t.key, t.modifier && (xn(t.modifier), t.m = t.modifier, delete t.modifier);
      break;
    }
    case 5: {
      const t = e;
      t.i = t.index, delete t.index;
      break;
    }
    case 4: {
      const t = e;
      t.k = t.key, delete t.key;
      break;
    }
    default:
      throw zn(G.UNHANDLED_MINIFIER_NODE_TYPE, null, {
        domain: hg,
        args: [e.type]
      });
  }
  delete e.type;
}
const gg = "parser";
function bg(e, t) {
  const { filename: n, breakLineCode: r, needIndent: o } = t, i = t.location !== !1, s = {
    filename: n,
    code: "",
    column: 1,
    line: 1,
    offset: 0,
    map: void 0,
    breakLineCode: r,
    needIndent: o,
    indentLevel: 0
  };
  i && e.loc && (s.source = e.loc.source);
  const a = () => s;
  function l(E, w) {
    s.code += E;
  }
  function u(E, w = !0) {
    const P = w ? r : "";
    l(o ? P + "  ".repeat(E) : P);
  }
  function c(E = !0) {
    const w = ++s.indentLevel;
    E && u(w);
  }
  function d(E = !0) {
    const w = --s.indentLevel;
    E && u(w);
  }
  function f() {
    u(s.indentLevel);
  }
  return {
    context: a,
    push: l,
    indent: c,
    deindent: d,
    newline: f,
    helper: (E) => `_${E}`,
    needIndent: () => s.needIndent
  };
}
function vg(e, t) {
  const { helper: n } = e;
  e.push(`${n(
    "linked"
    /* HelperNameMap.LINKED */
  )}(`), Kn(e, t.key), t.modifier ? (e.push(", "), Kn(e, t.modifier), e.push(", _type")) : e.push(", undefined, _type"), e.push(")");
}
function yg(e, t) {
  const { helper: n, needIndent: r } = e;
  e.push(`${n(
    "normalize"
    /* HelperNameMap.NORMALIZE */
  )}([`), e.indent(r());
  const o = t.items.length;
  for (let i = 0; i < o && (Kn(e, t.items[i]), i !== o - 1); i++)
    e.push(", ");
  e.deindent(r()), e.push("])");
}
function _g(e, t) {
  const { helper: n, needIndent: r } = e;
  if (t.cases.length > 1) {
    e.push(`${n(
      "plural"
      /* HelperNameMap.PLURAL */
    )}([`), e.indent(r());
    const o = t.cases.length;
    for (let i = 0; i < o && (Kn(e, t.cases[i]), i !== o - 1); i++)
      e.push(", ");
    e.deindent(r()), e.push("])");
  }
}
function Sg(e, t) {
  t.body ? Kn(e, t.body) : e.push("null");
}
function Kn(e, t) {
  const { helper: n } = e;
  switch (t.type) {
    case 0:
      Sg(e, t);
      break;
    case 1:
      _g(e, t);
      break;
    case 2:
      yg(e, t);
      break;
    case 6:
      vg(e, t);
      break;
    case 8:
      e.push(JSON.stringify(t.value), t);
      break;
    case 7:
      e.push(JSON.stringify(t.value), t);
      break;
    case 5:
      e.push(`${n(
        "interpolate"
        /* HelperNameMap.INTERPOLATE */
      )}(${n(
        "list"
        /* HelperNameMap.LIST */
      )}(${t.index}))`, t);
      break;
    case 4:
      e.push(`${n(
        "interpolate"
        /* HelperNameMap.INTERPOLATE */
      )}(${n(
        "named"
        /* HelperNameMap.NAMED */
      )}(${JSON.stringify(t.key)}))`, t);
      break;
    case 9:
      e.push(JSON.stringify(t.value), t);
      break;
    case 3:
      e.push(JSON.stringify(t.value), t);
      break;
    default:
      throw zn(G.UNHANDLED_CODEGEN_NODE_TYPE, null, {
        domain: gg,
        args: [t.type]
      });
  }
}
const Eg = (e, t = {}) => {
  const n = Da(t.mode) ? t.mode : "normal", r = Da(t.filename) ? t.filename : "message.intl";
  t.sourceMap;
  const o = t.breakLineCode != null ? t.breakLineCode : n === "arrow" ? ";" : `
`, i = t.needIndent ? t.needIndent : n !== "arrow", s = e.helpers || [], a = bg(e, {
    filename: r,
    breakLineCode: o,
    needIndent: i
  });
  a.push(n === "normal" ? "function __msg__ (ctx) {" : "(ctx) => {"), a.indent(i), s.length > 0 && (a.push(`const { ${dc(s.map((c) => `${c}: _${c}`), ", ")} } = ctx`), a.newline()), a.push("return "), Kn(a, e), a.deindent(i), a.push("}"), delete e.helpers;
  const { code: l, map: u } = a.context();
  return {
    ast: e,
    code: l,
    map: u ? u.toJSON() : void 0
    // eslint-disable-line @typescript-eslint/no-explicit-any
  };
};
function Tg(e, t = {}) {
  const n = cc({}, t), r = !!n.jit, o = !!n.minify, i = n.optimize == null ? !0 : n.optimize, a = dg(n).parse(e);
  return r ? (i && mg(a), o && xn(a), { ast: a, code: "" }) : (pg(a, n), Eg(a, n));
}
/*!
  * core-base v9.14.5
  * (c) 2025 kazuya kawaguchi
  * Released under the MIT License.
  */
function Cg() {
  typeof __INTLIFY_PROD_DEVTOOLS__ != "boolean" && (Gt().__INTLIFY_PROD_DEVTOOLS__ = !1), typeof __INTLIFY_JIT_COMPILATION__ != "boolean" && (Gt().__INTLIFY_JIT_COMPILATION__ = !1), typeof __INTLIFY_DROP_MESSAGE_COMPILER__ != "boolean" && (Gt().__INTLIFY_DROP_MESSAGE_COMPILER__ = !1);
}
function Mt(e) {
  return se(e) && os(e) === 0 && (_t(e, "b") || _t(e, "body"));
}
const fc = ["b", "body"];
function Og(e) {
  return fn(e, fc);
}
const pc = ["c", "cases"];
function Lg(e) {
  return fn(e, pc, []);
}
const mc = ["s", "static"];
function Pg(e) {
  return fn(e, mc);
}
const hc = ["i", "items"];
function wg(e) {
  return fn(e, hc, []);
}
const gc = ["t", "type"];
function os(e) {
  return fn(e, gc);
}
const bc = ["v", "value"];
function eo(e, t) {
  const n = fn(e, bc);
  if (n != null)
    return n;
  throw Rr(t);
}
const vc = ["m", "modifier"];
function kg(e) {
  return fn(e, vc);
}
const yc = ["k", "key"];
function $g(e) {
  const t = fn(e, yc);
  if (t)
    return t;
  throw Rr(
    6
    /* NodeTypes.Linked */
  );
}
function fn(e, t, n) {
  for (let r = 0; r < t.length; r++) {
    const o = t[r];
    if (_t(e, o) && e[o] != null)
      return e[o];
  }
  return n;
}
const _c = [
  ...fc,
  ...pc,
  ...mc,
  ...hc,
  ...yc,
  ...vc,
  ...bc,
  ...gc
];
function Rr(e) {
  return new Error(`unhandled node type: ${e}`);
}
const pn = [];
pn[
  0
  /* States.BEFORE_PATH */
] = {
  w: [
    0
    /* States.BEFORE_PATH */
  ],
  i: [
    3,
    0
    /* Actions.APPEND */
  ],
  "[": [
    4
    /* States.IN_SUB_PATH */
  ],
  o: [
    7
    /* States.AFTER_PATH */
  ]
};
pn[
  1
  /* States.IN_PATH */
] = {
  w: [
    1
    /* States.IN_PATH */
  ],
  ".": [
    2
    /* States.BEFORE_IDENT */
  ],
  "[": [
    4
    /* States.IN_SUB_PATH */
  ],
  o: [
    7
    /* States.AFTER_PATH */
  ]
};
pn[
  2
  /* States.BEFORE_IDENT */
] = {
  w: [
    2
    /* States.BEFORE_IDENT */
  ],
  i: [
    3,
    0
    /* Actions.APPEND */
  ],
  0: [
    3,
    0
    /* Actions.APPEND */
  ]
};
pn[
  3
  /* States.IN_IDENT */
] = {
  i: [
    3,
    0
    /* Actions.APPEND */
  ],
  0: [
    3,
    0
    /* Actions.APPEND */
  ],
  w: [
    1,
    1
    /* Actions.PUSH */
  ],
  ".": [
    2,
    1
    /* Actions.PUSH */
  ],
  "[": [
    4,
    1
    /* Actions.PUSH */
  ],
  o: [
    7,
    1
    /* Actions.PUSH */
  ]
};
pn[
  4
  /* States.IN_SUB_PATH */
] = {
  "'": [
    5,
    0
    /* Actions.APPEND */
  ],
  '"': [
    6,
    0
    /* Actions.APPEND */
  ],
  "[": [
    4,
    2
    /* Actions.INC_SUB_PATH_DEPTH */
  ],
  "]": [
    1,
    3
    /* Actions.PUSH_SUB_PATH */
  ],
  o: 8,
  l: [
    4,
    0
    /* Actions.APPEND */
  ]
};
pn[
  5
  /* States.IN_SINGLE_QUOTE */
] = {
  "'": [
    4,
    0
    /* Actions.APPEND */
  ],
  o: 8,
  l: [
    5,
    0
    /* Actions.APPEND */
  ]
};
pn[
  6
  /* States.IN_DOUBLE_QUOTE */
] = {
  '"': [
    4,
    0
    /* Actions.APPEND */
  ],
  o: 8,
  l: [
    6,
    0
    /* Actions.APPEND */
  ]
};
const Ng = /^\s?(?:true|false|-?[\d.]+|'[^']*'|"[^"]*")\s?$/;
function Ig(e) {
  return Ng.test(e);
}
function Ag(e) {
  const t = e.charCodeAt(0), n = e.charCodeAt(e.length - 1);
  return t === n && (t === 34 || t === 39) ? e.slice(1, -1) : e;
}
function xg(e) {
  if (e == null)
    return "o";
  switch (e.charCodeAt(0)) {
    case 91:
    // [
    case 93:
    // ]
    case 46:
    // .
    case 34:
    // "
    case 39:
      return e;
    case 95:
    // _
    case 36:
    // $
    case 45:
      return "i";
    case 9:
    // Tab (HT)
    case 10:
    // Newline (LF)
    case 13:
    // Return (CR)
    case 160:
    // No-break space (NBSP)
    case 65279:
    // Byte Order Mark (BOM)
    case 8232:
    // Line Separator (LS)
    case 8233:
      return "w";
  }
  return "i";
}
function Dg(e) {
  const t = e.trim();
  return e.charAt(0) === "0" && isNaN(parseInt(e)) ? !1 : Ig(t) ? Ag(t) : "*" + t;
}
function Rg(e) {
  const t = [];
  let n = -1, r = 0, o = 0, i, s, a, l, u, c, d;
  const f = [];
  f[
    0
    /* Actions.APPEND */
  ] = () => {
    s === void 0 ? s = a : s += a;
  }, f[
    1
    /* Actions.PUSH */
  ] = () => {
    s !== void 0 && (t.push(s), s = void 0);
  }, f[
    2
    /* Actions.INC_SUB_PATH_DEPTH */
  ] = () => {
    f[
      0
      /* Actions.APPEND */
    ](), o++;
  }, f[
    3
    /* Actions.PUSH_SUB_PATH */
  ] = () => {
    if (o > 0)
      o--, r = 4, f[
        0
        /* Actions.APPEND */
      ]();
    else {
      if (o = 0, s === void 0 || (s = Dg(s), s === !1))
        return !1;
      f[
        1
        /* Actions.PUSH */
      ]();
    }
  };
  function h() {
    const _ = e[n + 1];
    if (r === 5 && _ === "'" || r === 6 && _ === '"')
      return n++, a = "\\" + _, f[
        0
        /* Actions.APPEND */
      ](), !0;
  }
  for (; r !== null; )
    if (n++, i = e[n], !(i === "\\" && h())) {
      if (l = xg(i), d = pn[r], u = d[l] || d.l || 8, u === 8 || (r = u[0], u[1] !== void 0 && (c = f[u[1]], c && (a = i, c() === !1))))
        return;
      if (r === 7)
        return t;
    }
}
const ja = /* @__PURE__ */ new Map();
function Mg(e, t) {
  return se(e) ? e[t] : null;
}
function Fg(e, t) {
  if (!se(e))
    return null;
  let n = ja.get(t);
  if (n || (n = Rg(t), n && ja.set(t, n)), !n)
    return null;
  const r = n.length;
  let o = e, i = 0;
  for (; i < r; ) {
    const s = n[i];
    if (_c.includes(s) && Mt(o))
      return null;
    const a = o[s];
    if (a === void 0 || ve(o))
      return null;
    o = a, i++;
  }
  return o;
}
const jg = (e) => e, Ug = (e) => "", Vg = "text", Hg = (e) => e.length === 0 ? "" : zh(e), Wg = Yh;
function Ua(e, t) {
  return e = Math.abs(e), t === 2 ? e ? e > 1 ? 1 : 0 : 1 : e ? Math.min(e, 2) : 0;
}
function Bg(e) {
  const t = Le(e.pluralIndex) ? e.pluralIndex : -1;
  return e.named && (Le(e.named.count) || Le(e.named.n)) ? Le(e.named.count) ? e.named.count : Le(e.named.n) ? e.named.n : t : t;
}
function Kg(e, t) {
  t.count || (t.count = e), t.n || (t.n = e);
}
function Yg(e = {}) {
  const t = e.locale, n = Bg(e), r = se(e.pluralRules) && W(t) && ve(e.pluralRules[t]) ? e.pluralRules[t] : Ua, o = se(e.pluralRules) && W(t) && ve(e.pluralRules[t]) ? Ua : void 0, i = (P) => P[r(n, P.length, o)], s = e.list || [], a = (P) => s[P], l = e.named || pe();
  Le(e.pluralIndex) && Kg(n, l);
  const u = (P) => l[P];
  function c(P) {
    const M = ve(e.messages) ? e.messages(P) : se(e.messages) ? e.messages[P] : !1;
    return M || (e.parent ? e.parent.message(P) : Ug);
  }
  const d = (P) => e.modifiers ? e.modifiers[P] : jg, f = J(e.processor) && ve(e.processor.normalize) ? e.processor.normalize : Hg, h = J(e.processor) && ve(e.processor.interpolate) ? e.processor.interpolate : Wg, _ = J(e.processor) && W(e.processor.type) ? e.processor.type : Vg, w = {
    list: a,
    named: u,
    plural: i,
    linked: (P, ...M) => {
      const [S, g] = M;
      let O = "text", L = "";
      M.length === 1 ? se(S) ? (L = S.modifier || L, O = S.type || O) : W(S) && (L = S || L) : M.length === 2 && (W(S) && (L = S || L), W(g) && (O = g || O));
      const A = c(P)(w), F = (
        // The message in vnode resolved with linked are returned as an array by processor.nomalize
        O === "vnode" && Se(A) && L ? A[0] : A
      );
      return L ? d(L)(F, O) : F;
    },
    message: c,
    type: _,
    interpolate: h,
    normalize: f,
    values: Fe(pe(), s, l)
  };
  return w;
}
let Mr = null;
function zg(e) {
  Mr = e;
}
function Gg(e, t, n) {
  Mr && Mr.emit("i18n:init", {
    timestamp: Date.now(),
    i18n: e,
    version: t,
    meta: n
  });
}
const Xg = /* @__PURE__ */ Jg(
  "function:translate"
  /* IntlifyDevToolsHooks.FunctionTranslate */
);
function Jg(e) {
  return (t) => Mr && Mr.emit(e, t);
}
const qg = ns.__EXTEND_POINT__, yn = Io(qg), Zg = {
  // 2
  FALLBACK_TO_TRANSLATE: yn(),
  // 3
  CANNOT_FORMAT_NUMBER: yn(),
  // 4
  FALLBACK_TO_NUMBER_FORMAT: yn(),
  // 5
  CANNOT_FORMAT_DATE: yn(),
  // 6
  FALLBACK_TO_DATE_FORMAT: yn(),
  // 7
  EXPERIMENTAL_CUSTOM_MESSAGE_COMPILER: yn(),
  // 8
  __EXTEND_POINT__: yn()
  // 9
}, Sc = G.__EXTEND_POINT__, _n = Io(Sc), St = {
  INVALID_ARGUMENT: Sc,
  // 17
  INVALID_DATE_ARGUMENT: _n(),
  // 18
  INVALID_ISO_DATE_ARGUMENT: _n(),
  // 19
  NOT_SUPPORT_NON_STRING_MESSAGE: _n(),
  // 20
  NOT_SUPPORT_LOCALE_PROMISE_VALUE: _n(),
  // 21
  NOT_SUPPORT_LOCALE_ASYNC_FUNCTION: _n(),
  // 22
  NOT_SUPPORT_LOCALE_TYPE: _n(),
  // 23
  __EXTEND_POINT__: _n()
  // 24
};
function xt(e) {
  return zn(e, null, void 0);
}
function is(e, t) {
  return t.locale != null ? Va(t.locale) : Va(e.locale);
}
let Zo;
function Va(e) {
  if (W(e))
    return e;
  if (ve(e)) {
    if (e.resolvedOnce && Zo != null)
      return Zo;
    if (e.constructor.name === "Function") {
      const t = e();
      if (Kh(t))
        throw xt(St.NOT_SUPPORT_LOCALE_PROMISE_VALUE);
      return Zo = t;
    } else
      throw xt(St.NOT_SUPPORT_LOCALE_ASYNC_FUNCTION);
  } else
    throw xt(St.NOT_SUPPORT_LOCALE_TYPE);
}
function Qg(e, t, n) {
  return [.../* @__PURE__ */ new Set([
    n,
    ...Se(t) ? t : se(t) ? Object.keys(t) : W(t) ? [t] : [n]
  ])];
}
function Ec(e, t, n) {
  const r = W(n) ? n : Yn, o = e;
  o.__localeChainCache || (o.__localeChainCache = /* @__PURE__ */ new Map());
  let i = o.__localeChainCache.get(r);
  if (!i) {
    i = [];
    let s = [n];
    for (; Se(s); )
      s = Ha(i, s, t);
    const a = Se(t) || !J(t) ? t : t.default ? t.default : null;
    s = W(a) ? [a] : a, Se(s) && Ha(i, s, !1), o.__localeChainCache.set(r, i);
  }
  return i;
}
function Ha(e, t, n) {
  let r = !0;
  for (let o = 0; o < t.length && ee(r); o++) {
    const i = t[o];
    W(i) && (r = eb(e, t[o], n));
  }
  return r;
}
function eb(e, t, n) {
  let r;
  const o = t.split("-");
  do {
    const i = o.join("-");
    r = tb(e, i, n), o.splice(-1, 1);
  } while (o.length && r === !0);
  return r;
}
function tb(e, t, n) {
  let r = !1;
  if (!e.includes(t) && (r = !0, t)) {
    r = t[t.length - 1] !== "!";
    const o = t.replace(/!/g, "");
    e.push(o), (Se(n) || J(n)) && n[o] && (r = n[o]);
  }
  return r;
}
const nb = "9.14.5", Ao = -1, Yn = "en-US", Wa = "", Ba = (e) => `${e.charAt(0).toLocaleUpperCase()}${e.substr(1)}`;
function rb() {
  return {
    upper: (e, t) => t === "text" && W(e) ? e.toUpperCase() : t === "vnode" && se(e) && "__v_isVNode" in e ? e.children.toUpperCase() : e,
    lower: (e, t) => t === "text" && W(e) ? e.toLowerCase() : t === "vnode" && se(e) && "__v_isVNode" in e ? e.children.toLowerCase() : e,
    capitalize: (e, t) => t === "text" && W(e) ? Ba(e) : t === "vnode" && se(e) && "__v_isVNode" in e ? Ba(e.children) : e
  };
}
let Tc;
function Ka(e) {
  Tc = e;
}
let Cc;
function ob(e) {
  Cc = e;
}
let Oc;
function ib(e) {
  Oc = e;
}
let Lc = null;
const sb = /* @__NO_SIDE_EFFECTS__ */ (e) => {
  Lc = e;
}, ab = /* @__NO_SIDE_EFFECTS__ */ () => Lc;
let Pc = null;
const Ya = (e) => {
  Pc = e;
}, lb = () => Pc;
let za = 0;
function ub(e = {}) {
  const t = ve(e.onWarn) ? e.onWarn : Fh, n = W(e.version) ? e.version : nb, r = W(e.locale) || ve(e.locale) ? e.locale : Yn, o = ve(r) ? Yn : r, i = Se(e.fallbackLocale) || J(e.fallbackLocale) || W(e.fallbackLocale) || e.fallbackLocale === !1 ? e.fallbackLocale : o, s = J(e.messages) ? e.messages : Qo(o), a = J(e.datetimeFormats) ? e.datetimeFormats : Qo(o), l = J(e.numberFormats) ? e.numberFormats : Qo(o), u = Fe(pe(), e.modifiers, rb()), c = e.pluralRules || pe(), d = ve(e.missing) ? e.missing : null, f = ee(e.missingWarn) || cn(e.missingWarn) ? e.missingWarn : !0, h = ee(e.fallbackWarn) || cn(e.fallbackWarn) ? e.fallbackWarn : !0, _ = !!e.fallbackFormat, E = !!e.unresolving, w = ve(e.postTranslation) ? e.postTranslation : null, P = J(e.processor) ? e.processor : null, M = ee(e.warnHtmlMessage) ? e.warnHtmlMessage : !0, S = !!e.escapeParameter, g = ve(e.messageCompiler) ? e.messageCompiler : Tc, O = ve(e.messageResolver) ? e.messageResolver : Cc || Mg, L = ve(e.localeFallbacker) ? e.localeFallbacker : Oc || Qg, A = se(e.fallbackContext) ? e.fallbackContext : void 0, F = e, $ = se(F.__datetimeFormatters) ? F.__datetimeFormatters : /* @__PURE__ */ new Map(), B = se(F.__numberFormatters) ? F.__numberFormatters : /* @__PURE__ */ new Map(), Y = se(F.__meta) ? F.__meta : {};
  za++;
  const R = {
    version: n,
    cid: za,
    locale: r,
    fallbackLocale: i,
    messages: s,
    modifiers: u,
    pluralRules: c,
    missing: d,
    missingWarn: f,
    fallbackWarn: h,
    fallbackFormat: _,
    unresolving: E,
    postTranslation: w,
    processor: P,
    warnHtmlMessage: M,
    escapeParameter: S,
    messageCompiler: g,
    messageResolver: O,
    localeFallbacker: L,
    fallbackContext: A,
    onWarn: t,
    __meta: Y
  };
  return R.datetimeFormats = a, R.numberFormats = l, R.__datetimeFormatters = $, R.__numberFormatters = B, __INTLIFY_PROD_DEVTOOLS__ && Gg(R, n, Y), R;
}
const Qo = (e) => ({ [e]: pe() });
function ss(e, t, n, r, o) {
  const { missing: i, onWarn: s } = e;
  if (i !== null) {
    const a = i(e, n, t, o);
    return W(a) ? a : t;
  } else
    return t;
}
function Zn(e, t, n) {
  const r = e;
  r.__localeChainCache = /* @__PURE__ */ new Map(), e.localeFallbacker(e, n, t);
}
function cb(e, t) {
  return e === t ? !1 : e.split("-")[0] === t.split("-")[0];
}
function db(e, t) {
  const n = t.indexOf(e);
  if (n === -1)
    return !1;
  for (let r = n + 1; r < t.length; r++)
    if (cb(e, t[r]))
      return !0;
  return !1;
}
function ei(e) {
  return (n) => fb(n, e);
}
function fb(e, t) {
  const n = Og(t);
  if (n == null)
    throw Rr(
      0
      /* NodeTypes.Resource */
    );
  if (os(n) === 1) {
    const i = Lg(n);
    return e.plural(i.reduce((s, a) => [
      ...s,
      Ga(e, a)
    ], []));
  } else
    return Ga(e, n);
}
function Ga(e, t) {
  const n = Pg(t);
  if (n != null)
    return e.type === "text" ? n : e.normalize([n]);
  {
    const r = wg(t).reduce((o, i) => [...o, _i(e, i)], []);
    return e.normalize(r);
  }
}
function _i(e, t) {
  const n = os(t);
  switch (n) {
    case 3:
      return eo(t, n);
    case 9:
      return eo(t, n);
    case 4: {
      const r = t;
      if (_t(r, "k") && r.k)
        return e.interpolate(e.named(r.k));
      if (_t(r, "key") && r.key)
        return e.interpolate(e.named(r.key));
      throw Rr(n);
    }
    case 5: {
      const r = t;
      if (_t(r, "i") && Le(r.i))
        return e.interpolate(e.list(r.i));
      if (_t(r, "index") && Le(r.index))
        return e.interpolate(e.list(r.index));
      throw Rr(n);
    }
    case 6: {
      const r = t, o = kg(r), i = $g(r);
      return e.linked(_i(e, i), o ? _i(e, o) : void 0, e.type);
    }
    case 7:
      return eo(t, n);
    case 8:
      return eo(t, n);
    default:
      throw new Error(`unhandled node on format message part: ${n}`);
  }
}
const wc = (e) => e;
let Mn = pe();
function kc(e, t = {}) {
  let n = !1;
  const r = t.onError || eg;
  return t.onError = (o) => {
    n = !0, r(o);
  }, { ...Tg(e, t), detectError: n };
}
const pb = /* @__NO_SIDE_EFFECTS__ */ (e, t) => {
  if (!W(e))
    throw xt(St.NOT_SUPPORT_NON_STRING_MESSAGE);
  {
    ee(t.warnHtmlMessage) && t.warnHtmlMessage;
    const r = (t.onCacheKey || wc)(e), o = Mn[r];
    if (o)
      return o;
    const { code: i, detectError: s } = kc(e, t), a = new Function(`return ${i}`)();
    return s ? a : Mn[r] = a;
  }
};
function mb(e, t) {
  if (__INTLIFY_JIT_COMPILATION__ && !__INTLIFY_DROP_MESSAGE_COMPILER__ && W(e)) {
    ee(t.warnHtmlMessage) && t.warnHtmlMessage;
    const r = (t.onCacheKey || wc)(e), o = Mn[r];
    if (o)
      return o;
    const { ast: i, detectError: s } = kc(e, {
      ...t,
      location: !1,
      jit: !0
    }), a = ei(i);
    return s ? a : Mn[r] = a;
  } else {
    const n = e.cacheKey;
    if (n) {
      const r = Mn[n];
      return r || (Mn[n] = ei(e));
    } else
      return ei(e);
  }
}
const Xa = () => "", ct = (e) => ve(e);
function Ja(e, ...t) {
  const { fallbackFormat: n, postTranslation: r, unresolving: o, messageCompiler: i, fallbackLocale: s, messages: a } = e, [l, u] = Si(...t), c = ee(u.missingWarn) ? u.missingWarn : e.missingWarn, d = ee(u.fallbackWarn) ? u.fallbackWarn : e.fallbackWarn, f = ee(u.escapeParameter) ? u.escapeParameter : e.escapeParameter, h = !!u.resolvedMessage, _ = W(u.default) || ee(u.default) ? ee(u.default) ? i ? l : () => l : u.default : n ? i ? l : () => l : "", E = n || _ !== "", w = is(e, u);
  f && hb(u);
  let [P, M, S] = h ? [
    l,
    w,
    a[w] || pe()
  ] : $c(e, l, w, s, d, c), g = P, O = l;
  if (!h && !(W(g) || Mt(g) || ct(g)) && E && (g = _, O = g), !h && (!(W(g) || Mt(g) || ct(g)) || !W(M)))
    return o ? Ao : l;
  let L = !1;
  const A = () => {
    L = !0;
  }, F = ct(g) ? g : Nc(e, l, M, g, O, A);
  if (L)
    return g;
  const $ = vb(e, M, S, u), B = Yg($), Y = gb(e, F, B);
  let R = r ? r(Y, l) : Y;
  if (f && W(R) && (R = Wh(R)), __INTLIFY_PROD_DEVTOOLS__) {
    const z = {
      timestamp: Date.now(),
      key: W(l) ? l : ct(g) ? g.key : "",
      locale: M || (ct(g) ? g.locale : ""),
      format: W(g) ? g : ct(g) ? g.source : "",
      message: R
    };
    z.meta = Fe({}, e.__meta, /* @__PURE__ */ ab() || {}), Xg(z);
  }
  return R;
}
function hb(e) {
  Se(e.list) ? e.list = e.list.map((t) => W(t) ? Aa(t) : t) : se(e.named) && Object.keys(e.named).forEach((t) => {
    W(e.named[t]) && (e.named[t] = Aa(e.named[t]));
  });
}
function $c(e, t, n, r, o, i) {
  const { messages: s, onWarn: a, messageResolver: l, localeFallbacker: u } = e, c = u(e, r, n);
  let d = pe(), f, h = null;
  const _ = "translate";
  for (let E = 0; E < c.length && (f = c[E], d = s[f] || pe(), (h = l(d, t)) === null && (h = d[t]), !(W(h) || Mt(h) || ct(h))); E++)
    if (!db(f, c)) {
      const w = ss(
        e,
        // eslint-disable-line @typescript-eslint/no-explicit-any
        t,
        f,
        i,
        _
      );
      w !== t && (h = w);
    }
  return [h, f, d];
}
function Nc(e, t, n, r, o, i) {
  const { messageCompiler: s, warnHtmlMessage: a } = e;
  if (ct(r)) {
    const u = r;
    return u.locale = u.locale || n, u.key = u.key || t, u;
  }
  if (s == null) {
    const u = () => r;
    return u.locale = n, u.key = t, u;
  }
  const l = s(r, bb(e, n, o, r, a, i));
  return l.locale = n, l.key = t, l.source = r, l;
}
function gb(e, t, n) {
  return t(n);
}
function Si(...e) {
  const [t, n, r] = e, o = pe();
  if (!W(t) && !Le(t) && !ct(t) && !Mt(t))
    throw xt(St.INVALID_ARGUMENT);
  const i = Le(t) ? String(t) : (ct(t), t);
  return Le(n) ? o.plural = n : W(n) ? o.default = n : J(n) && !No(n) ? o.named = n : Se(n) && (o.list = n), Le(r) ? o.plural = r : W(r) ? o.default = r : J(r) && Fe(o, r), [i, o];
}
function bb(e, t, n, r, o, i) {
  return {
    locale: t,
    key: n,
    warnHtmlMessage: o,
    onError: (s) => {
      throw i && i(s), s;
    },
    onCacheKey: (s) => jh(t, n, s)
  };
}
function vb(e, t, n, r) {
  const { modifiers: o, pluralRules: i, messageResolver: s, fallbackLocale: a, fallbackWarn: l, missingWarn: u, fallbackContext: c } = e, f = {
    locale: t,
    modifiers: o,
    pluralRules: i,
    messages: (h) => {
      let _ = s(n, h);
      if (_ == null && c) {
        const [, , E] = $c(c, h, t, a, l, u);
        _ = s(E, h);
      }
      if (W(_) || Mt(_)) {
        let E = !1;
        const P = Nc(e, h, t, _, h, () => {
          E = !0;
        });
        return E ? Xa : P;
      } else return ct(_) ? _ : Xa;
    }
  };
  return e.processor && (f.processor = e.processor), r.list && (f.list = r.list), r.named && (f.named = r.named), Le(r.plural) && (f.pluralIndex = r.plural), f;
}
function qa(e, ...t) {
  const { datetimeFormats: n, unresolving: r, fallbackLocale: o, onWarn: i, localeFallbacker: s } = e, { __datetimeFormatters: a } = e, [l, u, c, d] = Ei(...t), f = ee(c.missingWarn) ? c.missingWarn : e.missingWarn;
  ee(c.fallbackWarn) ? c.fallbackWarn : e.fallbackWarn;
  const h = !!c.part, _ = is(e, c), E = s(
    e,
    // eslint-disable-line @typescript-eslint/no-explicit-any
    o,
    _
  );
  if (!W(l) || l === "")
    return new Intl.DateTimeFormat(_, d).format(u);
  let w = {}, P, M = null;
  const S = "datetime format";
  for (let L = 0; L < E.length && (P = E[L], w = n[P] || {}, M = w[l], !J(M)); L++)
    ss(e, l, P, f, S);
  if (!J(M) || !W(P))
    return r ? Ao : l;
  let g = `${P}__${l}`;
  No(d) || (g = `${g}__${JSON.stringify(d)}`);
  let O = a.get(g);
  return O || (O = new Intl.DateTimeFormat(P, Fe({}, M, d)), a.set(g, O)), h ? O.formatToParts(u) : O.format(u);
}
const Ic = [
  "localeMatcher",
  "weekday",
  "era",
  "year",
  "month",
  "day",
  "hour",
  "minute",
  "second",
  "timeZoneName",
  "formatMatcher",
  "hour12",
  "timeZone",
  "dateStyle",
  "timeStyle",
  "calendar",
  "dayPeriod",
  "numberingSystem",
  "hourCycle",
  "fractionalSecondDigits"
];
function Ei(...e) {
  const [t, n, r, o] = e, i = pe();
  let s = pe(), a;
  if (W(t)) {
    const l = t.match(/(\d{4}-\d{2}-\d{2})(T|\s)?(.*)/);
    if (!l)
      throw xt(St.INVALID_ISO_DATE_ARGUMENT);
    const u = l[3] ? l[3].trim().startsWith("T") ? `${l[1].trim()}${l[3].trim()}` : `${l[1].trim()}T${l[3].trim()}` : l[1].trim();
    a = new Date(u);
    try {
      a.toISOString();
    } catch {
      throw xt(St.INVALID_ISO_DATE_ARGUMENT);
    }
  } else if (Vh(t)) {
    if (isNaN(t.getTime()))
      throw xt(St.INVALID_DATE_ARGUMENT);
    a = t;
  } else if (Le(t))
    a = t;
  else
    throw xt(St.INVALID_ARGUMENT);
  return W(n) ? i.key = n : J(n) && Object.keys(n).forEach((l) => {
    Ic.includes(l) ? s[l] = n[l] : i[l] = n[l];
  }), W(r) ? i.locale = r : J(r) && (s = r), J(o) && (s = o), [i.key || "", a, i, s];
}
function Za(e, t, n) {
  const r = e;
  for (const o in n) {
    const i = `${t}__${o}`;
    r.__datetimeFormatters.has(i) && r.__datetimeFormatters.delete(i);
  }
}
function Qa(e, ...t) {
  const { numberFormats: n, unresolving: r, fallbackLocale: o, onWarn: i, localeFallbacker: s } = e, { __numberFormatters: a } = e, [l, u, c, d] = Ti(...t), f = ee(c.missingWarn) ? c.missingWarn : e.missingWarn;
  ee(c.fallbackWarn) ? c.fallbackWarn : e.fallbackWarn;
  const h = !!c.part, _ = is(e, c), E = s(
    e,
    // eslint-disable-line @typescript-eslint/no-explicit-any
    o,
    _
  );
  if (!W(l) || l === "")
    return new Intl.NumberFormat(_, d).format(u);
  let w = {}, P, M = null;
  const S = "number format";
  for (let L = 0; L < E.length && (P = E[L], w = n[P] || {}, M = w[l], !J(M)); L++)
    ss(e, l, P, f, S);
  if (!J(M) || !W(P))
    return r ? Ao : l;
  let g = `${P}__${l}`;
  No(d) || (g = `${g}__${JSON.stringify(d)}`);
  let O = a.get(g);
  return O || (O = new Intl.NumberFormat(P, Fe({}, M, d)), a.set(g, O)), h ? O.formatToParts(u) : O.format(u);
}
const Ac = [
  "localeMatcher",
  "style",
  "currency",
  "currencyDisplay",
  "currencySign",
  "useGrouping",
  "minimumIntegerDigits",
  "minimumFractionDigits",
  "maximumFractionDigits",
  "minimumSignificantDigits",
  "maximumSignificantDigits",
  "compactDisplay",
  "notation",
  "signDisplay",
  "unit",
  "unitDisplay",
  "roundingMode",
  "roundingPriority",
  "roundingIncrement",
  "trailingZeroDisplay"
];
function Ti(...e) {
  const [t, n, r, o] = e, i = pe();
  let s = pe();
  if (!Le(t))
    throw xt(St.INVALID_ARGUMENT);
  const a = t;
  return W(n) ? i.key = n : J(n) && Object.keys(n).forEach((l) => {
    Ac.includes(l) ? s[l] = n[l] : i[l] = n[l];
  }), W(r) ? i.locale = r : J(r) && (s = r), J(o) && (s = o), [i.key || "", a, i, s];
}
function el(e, t, n) {
  const r = e;
  for (const o in n) {
    const i = `${t}__${o}`;
    r.__numberFormatters.has(i) && r.__numberFormatters.delete(i);
  }
}
Cg();
/*!
  * vue-i18n v9.14.5
  * (c) 2025 kazuya kawaguchi
  * Released under the MIT License.
  */
const yb = "9.14.5";
function _b() {
  typeof __VUE_I18N_FULL_INSTALL__ != "boolean" && (Gt().__VUE_I18N_FULL_INSTALL__ = !0), typeof __VUE_I18N_LEGACY_API__ != "boolean" && (Gt().__VUE_I18N_LEGACY_API__ = !0), typeof __INTLIFY_JIT_COMPILATION__ != "boolean" && (Gt().__INTLIFY_JIT_COMPILATION__ = !1), typeof __INTLIFY_DROP_MESSAGE_COMPILER__ != "boolean" && (Gt().__INTLIFY_DROP_MESSAGE_COMPILER__ = !1), typeof __INTLIFY_PROD_DEVTOOLS__ != "boolean" && (Gt().__INTLIFY_PROD_DEVTOOLS__ = !1);
}
const Sb = Zg.__EXTEND_POINT__, Bt = Io(Sb);
Bt(), Bt(), Bt(), Bt(), Bt(), Bt(), Bt(), Bt(), Bt();
const xc = St.__EXTEND_POINT__, nt = Io(xc), $e = {
  // composer module errors
  UNEXPECTED_RETURN_TYPE: xc,
  // 24
  // legacy module errors
  INVALID_ARGUMENT: nt(),
  // 25
  // i18n module errors
  MUST_BE_CALL_SETUP_TOP: nt(),
  // 26
  NOT_INSTALLED: nt(),
  // 27
  NOT_AVAILABLE_IN_LEGACY_MODE: nt(),
  // 28
  // directive module errors
  REQUIRED_VALUE: nt(),
  // 29
  INVALID_VALUE: nt(),
  // 30
  // vue-devtools errors
  CANNOT_SETUP_VUE_DEVTOOLS_PLUGIN: nt(),
  // 31
  NOT_INSTALLED_WITH_PROVIDE: nt(),
  // 32
  // unexpected error
  UNEXPECTED_ERROR: nt(),
  // 33
  // not compatible legacy vue-i18n constructor
  NOT_COMPATIBLE_LEGACY_VUE_I18N: nt(),
  // 34
  // bridge support vue 2.x only
  BRIDGE_SUPPORT_VUE_2_ONLY: nt(),
  // 35
  // need to define `i18n` option in `allowComposition: true` and `useScope: 'local' at `useI18n``
  MUST_DEFINE_I18N_OPTION_IN_ALLOW_COMPOSITION: nt(),
  // 36
  // Not available Compostion API in Legacy API mode. Please make sure that the legacy API mode is working properly
  NOT_AVAILABLE_COMPOSITION_IN_LEGACY: nt(),
  // 37
  // for enhancement
  __EXTEND_POINT__: nt()
  // 38
};
function Ne(e, ...t) {
  return zn(e, null, void 0);
}
const Ci = /* @__PURE__ */ dn("__translateVNode"), Oi = /* @__PURE__ */ dn("__datetimeParts"), Li = /* @__PURE__ */ dn("__numberParts"), Dc = dn("__setPluralRules"), Rc = /* @__PURE__ */ dn("__injectWithOption"), Pi = /* @__PURE__ */ dn("__dispose");
function Fr(e) {
  if (!se(e) || Mt(e))
    return e;
  for (const t in e)
    if (_t(e, t))
      if (!t.includes("."))
        se(e[t]) && Fr(e[t]);
      else {
        const n = t.split("."), r = n.length - 1;
        let o = e, i = !1;
        for (let s = 0; s < r; s++) {
          if (n[s] === "__proto__")
            throw new Error(`unsafe key: ${n[s]}`);
          if (n[s] in o || (o[n[s]] = pe()), !se(o[n[s]])) {
            i = !0;
            break;
          }
          o = o[n[s]];
        }
        if (i || (Mt(o) ? _c.includes(n[r]) || delete e[t] : (o[n[r]] = e[t], delete e[t])), !Mt(o)) {
          const s = o[n[r]];
          se(s) && Fr(s);
        }
      }
  return e;
}
function xo(e, t) {
  const { messages: n, __i18n: r, messageResolver: o, flatJson: i } = t, s = J(n) ? n : Se(r) ? pe() : { [e]: pe() };
  if (Se(r) && r.forEach((a) => {
    if ("locale" in a && "resource" in a) {
      const { locale: l, resource: u } = a;
      l ? (s[l] = s[l] || pe(), io(u, s[l])) : io(u, s);
    } else
      W(a) && io(JSON.parse(a), s);
  }), o == null && i)
    for (const a in s)
      _t(s, a) && Fr(s[a]);
  return s;
}
function Mc(e) {
  return e.type;
}
function Fc(e, t, n) {
  let r = se(t.messages) ? t.messages : pe();
  "__i18nGlobal" in n && (r = xo(e.locale.value, {
    messages: r,
    __i18n: n.__i18nGlobal
  }));
  const o = Object.keys(r);
  o.length && o.forEach((i) => {
    e.mergeLocaleMessage(i, r[i]);
  });
  {
    if (se(t.datetimeFormats)) {
      const i = Object.keys(t.datetimeFormats);
      i.length && i.forEach((s) => {
        e.mergeDateTimeFormat(s, t.datetimeFormats[s]);
      });
    }
    if (se(t.numberFormats)) {
      const i = Object.keys(t.numberFormats);
      i.length && i.forEach((s) => {
        e.mergeNumberFormat(s, t.numberFormats[s]);
      });
    }
  }
}
function tl(e) {
  return ke(Hr, null, e, 0);
}
const nl = "__INTLIFY_META__", rl = () => [], Eb = () => !1;
let ol = 0;
function il(e) {
  return (t, n, r, o) => e(n, r, jt() || void 0, o);
}
const Tb = /* @__NO_SIDE_EFFECTS__ */ () => {
  const e = jt();
  let t = null;
  return e && (t = Mc(e)[nl]) ? { [nl]: t } : null;
};
function as(e = {}, t) {
  const { __root: n, __injectWithOption: r } = e, o = n === void 0, i = e.flatJson, s = go ? De : Ml, a = !!e.translateExistCompatible;
  let l = ee(e.inheritLocale) ? e.inheritLocale : !0;
  const u = s(
    // prettier-ignore
    n && l ? n.locale.value : W(e.locale) ? e.locale : Yn
  ), c = s(
    // prettier-ignore
    n && l ? n.fallbackLocale.value : W(e.fallbackLocale) || Se(e.fallbackLocale) || J(e.fallbackLocale) || e.fallbackLocale === !1 ? e.fallbackLocale : u.value
  ), d = s(xo(u.value, e)), f = s(J(e.datetimeFormats) ? e.datetimeFormats : { [u.value]: {} }), h = s(J(e.numberFormats) ? e.numberFormats : { [u.value]: {} });
  let _ = n ? n.missingWarn : ee(e.missingWarn) || cn(e.missingWarn) ? e.missingWarn : !0, E = n ? n.fallbackWarn : ee(e.fallbackWarn) || cn(e.fallbackWarn) ? e.fallbackWarn : !0, w = n ? n.fallbackRoot : ee(e.fallbackRoot) ? e.fallbackRoot : !0, P = !!e.fallbackFormat, M = ve(e.missing) ? e.missing : null, S = ve(e.missing) ? il(e.missing) : null, g = ve(e.postTranslation) ? e.postTranslation : null, O = n ? n.warnHtmlMessage : ee(e.warnHtmlMessage) ? e.warnHtmlMessage : !0, L = !!e.escapeParameter;
  const A = n ? n.modifiers : J(e.modifiers) ? e.modifiers : {};
  let F = e.pluralRules || n && n.pluralRules, $;
  $ = (() => {
    o && Ya(null);
    const C = {
      version: yb,
      locale: u.value,
      fallbackLocale: c.value,
      messages: d.value,
      modifiers: A,
      pluralRules: F,
      missing: S === null ? void 0 : S,
      missingWarn: _,
      fallbackWarn: E,
      fallbackFormat: P,
      unresolving: !0,
      postTranslation: g === null ? void 0 : g,
      warnHtmlMessage: O,
      escapeParameter: L,
      messageResolver: e.messageResolver,
      messageCompiler: e.messageCompiler,
      __meta: { framework: "vue" }
    };
    C.datetimeFormats = f.value, C.numberFormats = h.value, C.__datetimeFormatters = J($) ? $.__datetimeFormatters : void 0, C.__numberFormatters = J($) ? $.__numberFormatters : void 0;
    const I = ub(C);
    return o && Ya(I), I;
  })(), Zn($, u.value, c.value);
  function Y() {
    return [
      u.value,
      c.value,
      d.value,
      f.value,
      h.value
    ];
  }
  const R = dt({
    get: () => u.value,
    set: (C) => {
      u.value = C, $.locale = u.value;
    }
  }), z = dt({
    get: () => c.value,
    set: (C) => {
      c.value = C, $.fallbackLocale = c.value, Zn($, u.value, C);
    }
  }), ae = dt(() => d.value), Te = /* @__PURE__ */ dt(() => f.value), ne = /* @__PURE__ */ dt(() => h.value);
  function te() {
    return ve(g) ? g : null;
  }
  function Q(C) {
    g = C, $.postTranslation = C;
  }
  function Pe() {
    return M;
  }
  function we(C) {
    C !== null && (S = il(C)), M = C, $.missing = S;
  }
  const ue = (C, I, K, re, Ce, Ke) => {
    Y();
    let Ae;
    try {
      __INTLIFY_PROD_DEVTOOLS__, o || ($.fallbackContext = n ? lb() : void 0), Ae = C($);
    } finally {
      __INTLIFY_PROD_DEVTOOLS__, o || ($.fallbackContext = void 0);
    }
    if (K !== "translate exists" && // for not `te` (e.g `t`)
    Le(Ae) && Ae === Ao || K === "translate exists" && !Ae) {
      const [mn, Do] = I();
      return n && w ? re(n) : Ce(mn);
    } else {
      if (Ke(Ae))
        return Ae;
      throw Ne($e.UNEXPECTED_RETURN_TYPE);
    }
  };
  function he(...C) {
    return ue((I) => Reflect.apply(Ja, null, [I, ...C]), () => Si(...C), "translate", (I) => Reflect.apply(I.t, I, [...C]), (I) => I, (I) => W(I));
  }
  function it(...C) {
    const [I, K, re] = C;
    if (re && !se(re))
      throw Ne($e.INVALID_ARGUMENT);
    return he(I, K, Fe({ resolvedMessage: !0 }, re || {}));
  }
  function Be(...C) {
    return ue((I) => Reflect.apply(qa, null, [I, ...C]), () => Ei(...C), "datetime format", (I) => Reflect.apply(I.d, I, [...C]), () => Wa, (I) => W(I));
  }
  function Lt(...C) {
    return ue((I) => Reflect.apply(Qa, null, [I, ...C]), () => Ti(...C), "number format", (I) => Reflect.apply(I.n, I, [...C]), () => Wa, (I) => W(I));
  }
  function Qe(C) {
    return C.map((I) => W(I) || Le(I) || ee(I) ? tl(String(I)) : I);
  }
  const ht = {
    normalize: Qe,
    interpolate: (C) => C,
    type: "vnode"
  };
  function Pt(...C) {
    return ue(
      (I) => {
        let K;
        const re = I;
        try {
          re.processor = ht, K = Reflect.apply(Ja, null, [re, ...C]);
        } finally {
          re.processor = null;
        }
        return K;
      },
      () => Si(...C),
      "translate",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (I) => I[Ci](...C),
      (I) => [tl(I)],
      (I) => Se(I)
    );
  }
  function et(...C) {
    return ue(
      (I) => Reflect.apply(Qa, null, [I, ...C]),
      () => Ti(...C),
      "number format",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (I) => I[Li](...C),
      rl,
      (I) => W(I) || Se(I)
    );
  }
  function Qt(...C) {
    return ue(
      (I) => Reflect.apply(qa, null, [I, ...C]),
      () => Ei(...C),
      "datetime format",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (I) => I[Oi](...C),
      rl,
      (I) => W(I) || Se(I)
    );
  }
  function b(C) {
    F = C, $.pluralRules = F;
  }
  function y(C, I) {
    return ue(() => {
      if (!C)
        return !1;
      const K = W(I) ? I : u.value, re = x(K), Ce = $.messageResolver(re, C);
      return a ? Ce != null : Mt(Ce) || ct(Ce) || W(Ce);
    }, () => [C], "translate exists", (K) => Reflect.apply(K.te, K, [C, I]), Eb, (K) => ee(K));
  }
  function v(C) {
    let I = null;
    const K = Ec($, c.value, u.value);
    for (let re = 0; re < K.length; re++) {
      const Ce = d.value[K[re]] || {}, Ke = $.messageResolver(Ce, C);
      if (Ke != null) {
        I = Ke;
        break;
      }
    }
    return I;
  }
  function N(C) {
    const I = v(C);
    return I ?? (n ? n.tm(C) || {} : {});
  }
  function x(C) {
    return d.value[C] || {};
  }
  function D(C, I) {
    if (i) {
      const K = { [C]: I };
      for (const re in K)
        _t(K, re) && Fr(K[re]);
      I = K[C];
    }
    d.value[C] = I, $.messages = d.value;
  }
  function V(C, I) {
    d.value[C] = d.value[C] || {};
    const K = { [C]: I };
    if (i)
      for (const re in K)
        _t(K, re) && Fr(K[re]);
    I = K[C], io(I, d.value[C]), $.messages = d.value;
  }
  function U(C) {
    return f.value[C] || {};
  }
  function p(C, I) {
    f.value[C] = I, $.datetimeFormats = f.value, Za($, C, I);
  }
  function m(C, I) {
    f.value[C] = Fe(f.value[C] || {}, I), $.datetimeFormats = f.value, Za($, C, I);
  }
  function T(C) {
    return h.value[C] || {};
  }
  function k(C, I) {
    h.value[C] = I, $.numberFormats = h.value, el($, C, I);
  }
  function H(C, I) {
    h.value[C] = Fe(h.value[C] || {}, I), $.numberFormats = h.value, el($, C, I);
  }
  ol++, n && go && (ft(n.locale, (C) => {
    l && (u.value = C, $.locale = C, Zn($, u.value, c.value));
  }), ft(n.fallbackLocale, (C) => {
    l && (c.value = C, $.fallbackLocale = C, Zn($, u.value, c.value));
  }));
  const j = {
    id: ol,
    locale: R,
    fallbackLocale: z,
    get inheritLocale() {
      return l;
    },
    set inheritLocale(C) {
      l = C, C && n && (u.value = n.locale.value, c.value = n.fallbackLocale.value, Zn($, u.value, c.value));
    },
    get availableLocales() {
      return Object.keys(d.value).sort();
    },
    messages: ae,
    get modifiers() {
      return A;
    },
    get pluralRules() {
      return F || {};
    },
    get isGlobal() {
      return o;
    },
    get missingWarn() {
      return _;
    },
    set missingWarn(C) {
      _ = C, $.missingWarn = _;
    },
    get fallbackWarn() {
      return E;
    },
    set fallbackWarn(C) {
      E = C, $.fallbackWarn = E;
    },
    get fallbackRoot() {
      return w;
    },
    set fallbackRoot(C) {
      w = C;
    },
    get fallbackFormat() {
      return P;
    },
    set fallbackFormat(C) {
      P = C, $.fallbackFormat = P;
    },
    get warnHtmlMessage() {
      return O;
    },
    set warnHtmlMessage(C) {
      O = C, $.warnHtmlMessage = C;
    },
    get escapeParameter() {
      return L;
    },
    set escapeParameter(C) {
      L = C, $.escapeParameter = C;
    },
    t: he,
    getLocaleMessage: x,
    setLocaleMessage: D,
    mergeLocaleMessage: V,
    getPostTranslationHandler: te,
    setPostTranslationHandler: Q,
    getMissingHandler: Pe,
    setMissingHandler: we,
    [Dc]: b
  };
  return j.datetimeFormats = Te, j.numberFormats = ne, j.rt = it, j.te = y, j.tm = N, j.d = Be, j.n = Lt, j.getDateTimeFormat = U, j.setDateTimeFormat = p, j.mergeDateTimeFormat = m, j.getNumberFormat = T, j.setNumberFormat = k, j.mergeNumberFormat = H, j[Rc] = r, j[Ci] = Pt, j[Oi] = Qt, j[Li] = et, j;
}
function Cb(e) {
  const t = W(e.locale) ? e.locale : Yn, n = W(e.fallbackLocale) || Se(e.fallbackLocale) || J(e.fallbackLocale) || e.fallbackLocale === !1 ? e.fallbackLocale : t, r = ve(e.missing) ? e.missing : void 0, o = ee(e.silentTranslationWarn) || cn(e.silentTranslationWarn) ? !e.silentTranslationWarn : !0, i = ee(e.silentFallbackWarn) || cn(e.silentFallbackWarn) ? !e.silentFallbackWarn : !0, s = ee(e.fallbackRoot) ? e.fallbackRoot : !0, a = !!e.formatFallbackMessages, l = J(e.modifiers) ? e.modifiers : {}, u = e.pluralizationRules, c = ve(e.postTranslation) ? e.postTranslation : void 0, d = W(e.warnHtmlInMessage) ? e.warnHtmlInMessage !== "off" : !0, f = !!e.escapeParameterHtml, h = ee(e.sync) ? e.sync : !0;
  let _ = e.messages;
  if (J(e.sharedMessages)) {
    const L = e.sharedMessages;
    _ = Object.keys(L).reduce((F, $) => {
      const B = F[$] || (F[$] = {});
      return Fe(B, L[$]), F;
    }, _ || {});
  }
  const { __i18n: E, __root: w, __injectWithOption: P } = e, M = e.datetimeFormats, S = e.numberFormats, g = e.flatJson, O = e.translateExistCompatible;
  return {
    locale: t,
    fallbackLocale: n,
    messages: _,
    flatJson: g,
    datetimeFormats: M,
    numberFormats: S,
    missing: r,
    missingWarn: o,
    fallbackWarn: i,
    fallbackRoot: s,
    fallbackFormat: a,
    modifiers: l,
    pluralRules: u,
    postTranslation: c,
    warnHtmlMessage: d,
    escapeParameter: f,
    messageResolver: e.messageResolver,
    inheritLocale: h,
    translateExistCompatible: O,
    __i18n: E,
    __root: w,
    __injectWithOption: P
  };
}
function wi(e = {}, t) {
  {
    const n = as(Cb(e)), { __extender: r } = e, o = {
      // id
      id: n.id,
      // locale
      get locale() {
        return n.locale.value;
      },
      set locale(i) {
        n.locale.value = i;
      },
      // fallbackLocale
      get fallbackLocale() {
        return n.fallbackLocale.value;
      },
      set fallbackLocale(i) {
        n.fallbackLocale.value = i;
      },
      // messages
      get messages() {
        return n.messages.value;
      },
      // datetimeFormats
      get datetimeFormats() {
        return n.datetimeFormats.value;
      },
      // numberFormats
      get numberFormats() {
        return n.numberFormats.value;
      },
      // availableLocales
      get availableLocales() {
        return n.availableLocales;
      },
      // formatter
      get formatter() {
        return {
          interpolate() {
            return [];
          }
        };
      },
      set formatter(i) {
      },
      // missing
      get missing() {
        return n.getMissingHandler();
      },
      set missing(i) {
        n.setMissingHandler(i);
      },
      // silentTranslationWarn
      get silentTranslationWarn() {
        return ee(n.missingWarn) ? !n.missingWarn : n.missingWarn;
      },
      set silentTranslationWarn(i) {
        n.missingWarn = ee(i) ? !i : i;
      },
      // silentFallbackWarn
      get silentFallbackWarn() {
        return ee(n.fallbackWarn) ? !n.fallbackWarn : n.fallbackWarn;
      },
      set silentFallbackWarn(i) {
        n.fallbackWarn = ee(i) ? !i : i;
      },
      // modifiers
      get modifiers() {
        return n.modifiers;
      },
      // formatFallbackMessages
      get formatFallbackMessages() {
        return n.fallbackFormat;
      },
      set formatFallbackMessages(i) {
        n.fallbackFormat = i;
      },
      // postTranslation
      get postTranslation() {
        return n.getPostTranslationHandler();
      },
      set postTranslation(i) {
        n.setPostTranslationHandler(i);
      },
      // sync
      get sync() {
        return n.inheritLocale;
      },
      set sync(i) {
        n.inheritLocale = i;
      },
      // warnInHtmlMessage
      get warnHtmlInMessage() {
        return n.warnHtmlMessage ? "warn" : "off";
      },
      set warnHtmlInMessage(i) {
        n.warnHtmlMessage = i !== "off";
      },
      // escapeParameterHtml
      get escapeParameterHtml() {
        return n.escapeParameter;
      },
      set escapeParameterHtml(i) {
        n.escapeParameter = i;
      },
      // preserveDirectiveContent
      get preserveDirectiveContent() {
        return !0;
      },
      set preserveDirectiveContent(i) {
      },
      // pluralizationRules
      get pluralizationRules() {
        return n.pluralRules || {};
      },
      // for internal
      __composer: n,
      // t
      t(...i) {
        const [s, a, l] = i, u = {};
        let c = null, d = null;
        if (!W(s))
          throw Ne($e.INVALID_ARGUMENT);
        const f = s;
        return W(a) ? u.locale = a : Se(a) ? c = a : J(a) && (d = a), Se(l) ? c = l : J(l) && (d = l), Reflect.apply(n.t, n, [
          f,
          c || d || {},
          u
        ]);
      },
      rt(...i) {
        return Reflect.apply(n.rt, n, [...i]);
      },
      // tc
      tc(...i) {
        const [s, a, l] = i, u = { plural: 1 };
        let c = null, d = null;
        if (!W(s))
          throw Ne($e.INVALID_ARGUMENT);
        const f = s;
        return W(a) ? u.locale = a : Le(a) ? u.plural = a : Se(a) ? c = a : J(a) && (d = a), W(l) ? u.locale = l : Se(l) ? c = l : J(l) && (d = l), Reflect.apply(n.t, n, [
          f,
          c || d || {},
          u
        ]);
      },
      // te
      te(i, s) {
        return n.te(i, s);
      },
      // tm
      tm(i) {
        return n.tm(i);
      },
      // getLocaleMessage
      getLocaleMessage(i) {
        return n.getLocaleMessage(i);
      },
      // setLocaleMessage
      setLocaleMessage(i, s) {
        n.setLocaleMessage(i, s);
      },
      // mergeLocaleMessage
      mergeLocaleMessage(i, s) {
        n.mergeLocaleMessage(i, s);
      },
      // d
      d(...i) {
        return Reflect.apply(n.d, n, [...i]);
      },
      // getDateTimeFormat
      getDateTimeFormat(i) {
        return n.getDateTimeFormat(i);
      },
      // setDateTimeFormat
      setDateTimeFormat(i, s) {
        n.setDateTimeFormat(i, s);
      },
      // mergeDateTimeFormat
      mergeDateTimeFormat(i, s) {
        n.mergeDateTimeFormat(i, s);
      },
      // n
      n(...i) {
        return Reflect.apply(n.n, n, [...i]);
      },
      // getNumberFormat
      getNumberFormat(i) {
        return n.getNumberFormat(i);
      },
      // setNumberFormat
      setNumberFormat(i, s) {
        n.setNumberFormat(i, s);
      },
      // mergeNumberFormat
      mergeNumberFormat(i, s) {
        n.mergeNumberFormat(i, s);
      },
      // getChoiceIndex
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      getChoiceIndex(i, s) {
        return -1;
      }
    };
    return o.__extender = r, o;
  }
}
const ls = {
  tag: {
    type: [String, Object]
  },
  locale: {
    type: String
  },
  scope: {
    type: String,
    // NOTE: avoid https://github.com/microsoft/rushstack/issues/1050
    validator: (e) => e === "parent" || e === "global",
    default: "parent"
    /* ComponentI18nScope */
  },
  i18n: {
    type: Object
  }
};
function Ob({ slots: e }, t) {
  return t.length === 1 && t[0] === "default" ? (e.default ? e.default() : []).reduce((r, o) => [
    ...r,
    // prettier-ignore
    ...o.type === Ve ? o.children : [o]
  ], []) : t.reduce((n, r) => {
    const o = e[r];
    return o && (n[r] = o()), n;
  }, pe());
}
function jc(e) {
  return Ve;
}
const Lb = /* @__PURE__ */ Ur({
  /* eslint-disable */
  name: "i18n-t",
  props: Fe({
    keypath: {
      type: String,
      required: !0
    },
    plural: {
      type: [Number, String],
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      validator: (e) => Le(e) || !isNaN(e)
    }
  }, ls),
  /* eslint-enable */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setup(e, t) {
    const { slots: n, attrs: r } = t, o = e.i18n || us({
      useScope: e.scope,
      __useComponent: !0
    });
    return () => {
      const i = Object.keys(n).filter((d) => d !== "_"), s = pe();
      e.locale && (s.locale = e.locale), e.plural !== void 0 && (s.plural = W(e.plural) ? +e.plural : e.plural);
      const a = Ob(t, i), l = o[Ci](e.keypath, a, s), u = Fe(pe(), r), c = W(e.tag) || se(e.tag) ? e.tag : jc();
      return Qi(c, u, l);
    };
  }
}), sl = Lb;
function Pb(e) {
  return Se(e) && !W(e[0]);
}
function Uc(e, t, n, r) {
  const { slots: o, attrs: i } = t;
  return () => {
    const s = { part: !0 };
    let a = pe();
    e.locale && (s.locale = e.locale), W(e.format) ? s.key = e.format : se(e.format) && (W(e.format.key) && (s.key = e.format.key), a = Object.keys(e.format).reduce((f, h) => n.includes(h) ? Fe(pe(), f, { [h]: e.format[h] }) : f, pe()));
    const l = r(e.value, s, a);
    let u = [s.key];
    Se(l) ? u = l.map((f, h) => {
      const _ = o[f.type], E = _ ? _({ [f.type]: f.value, index: h, parts: l }) : [f.value];
      return Pb(E) && (E[0].key = `${f.type}-${h}`), E;
    }) : W(l) && (u = [l]);
    const c = Fe(pe(), i), d = W(e.tag) || se(e.tag) ? e.tag : jc();
    return Qi(d, c, u);
  };
}
const wb = /* @__PURE__ */ Ur({
  /* eslint-disable */
  name: "i18n-n",
  props: Fe({
    value: {
      type: Number,
      required: !0
    },
    format: {
      type: [String, Object]
    }
  }, ls),
  /* eslint-enable */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setup(e, t) {
    const n = e.i18n || us({
      useScope: e.scope,
      __useComponent: !0
    });
    return Uc(e, t, Ac, (...r) => (
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      n[Li](...r)
    ));
  }
}), al = wb, kb = /* @__PURE__ */ Ur({
  /* eslint-disable */
  name: "i18n-d",
  props: Fe({
    value: {
      type: [Number, Date],
      required: !0
    },
    format: {
      type: [String, Object]
    }
  }, ls),
  /* eslint-enable */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setup(e, t) {
    const n = e.i18n || us({
      useScope: e.scope,
      __useComponent: !0
    });
    return Uc(e, t, Ic, (...r) => (
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      n[Oi](...r)
    ));
  }
}), ll = kb;
function $b(e, t) {
  const n = e;
  if (e.mode === "composition")
    return n.__getInstance(t) || e.global;
  {
    const r = n.__getInstance(t);
    return r != null ? r.__composer : e.global.__composer;
  }
}
function Nb(e) {
  const t = (s) => {
    const { instance: a, modifiers: l, value: u } = s;
    if (!a || !a.$)
      throw Ne($e.UNEXPECTED_ERROR);
    const c = $b(e, a.$), d = ul(u);
    return [
      Reflect.apply(c.t, c, [...cl(d)]),
      c
    ];
  };
  return {
    created: (s, a) => {
      const [l, u] = t(a);
      go && e.global === u && (s.__i18nWatcher = ft(u.locale, () => {
        a.instance && a.instance.$forceUpdate();
      })), s.__composer = u, s.textContent = l;
    },
    unmounted: (s) => {
      go && s.__i18nWatcher && (s.__i18nWatcher(), s.__i18nWatcher = void 0, delete s.__i18nWatcher), s.__composer && (s.__composer = void 0, delete s.__composer);
    },
    beforeUpdate: (s, { value: a }) => {
      if (s.__composer) {
        const l = s.__composer, u = ul(a);
        s.textContent = Reflect.apply(l.t, l, [
          ...cl(u)
        ]);
      }
    },
    getSSRProps: (s) => {
      const [a] = t(s);
      return { textContent: a };
    }
  };
}
function ul(e) {
  if (W(e))
    return { path: e };
  if (J(e)) {
    if (!("path" in e))
      throw Ne($e.REQUIRED_VALUE, "path");
    return e;
  } else
    throw Ne($e.INVALID_VALUE);
}
function cl(e) {
  const { path: t, locale: n, args: r, choice: o, plural: i } = e, s = {}, a = r || {};
  return W(n) && (s.locale = n), Le(o) && (s.plural = o), Le(i) && (s.plural = i), [t, a, s];
}
function Ib(e, t, ...n) {
  const r = J(n[0]) ? n[0] : {}, o = !!r.useI18nComponentName;
  (ee(r.globalInstall) ? r.globalInstall : !0) && ([o ? "i18n" : sl.name, "I18nT"].forEach((s) => e.component(s, sl)), [al.name, "I18nN"].forEach((s) => e.component(s, al)), [ll.name, "I18nD"].forEach((s) => e.component(s, ll))), e.directive("t", Nb(t));
}
function Ab(e, t, n) {
  return {
    beforeCreate() {
      const r = jt();
      if (!r)
        throw Ne($e.UNEXPECTED_ERROR);
      const o = this.$options;
      if (o.i18n) {
        const i = o.i18n;
        if (o.__i18n && (i.__i18n = o.__i18n), i.__root = t, this === this.$root)
          this.$i18n = dl(e, i);
        else {
          i.__injectWithOption = !0, i.__extender = n.__vueI18nExtend, this.$i18n = wi(i);
          const s = this.$i18n;
          s.__extender && (s.__disposer = s.__extender(this.$i18n));
        }
      } else if (o.__i18n)
        if (this === this.$root)
          this.$i18n = dl(e, o);
        else {
          this.$i18n = wi({
            __i18n: o.__i18n,
            __injectWithOption: !0,
            __extender: n.__vueI18nExtend,
            __root: t
          });
          const i = this.$i18n;
          i.__extender && (i.__disposer = i.__extender(this.$i18n));
        }
      else
        this.$i18n = e;
      o.__i18nGlobal && Fc(t, o, o), this.$t = (...i) => this.$i18n.t(...i), this.$rt = (...i) => this.$i18n.rt(...i), this.$tc = (...i) => this.$i18n.tc(...i), this.$te = (i, s) => this.$i18n.te(i, s), this.$d = (...i) => this.$i18n.d(...i), this.$n = (...i) => this.$i18n.n(...i), this.$tm = (i) => this.$i18n.tm(i), n.__setInstance(r, this.$i18n);
    },
    mounted() {
    },
    unmounted() {
      const r = jt();
      if (!r)
        throw Ne($e.UNEXPECTED_ERROR);
      const o = this.$i18n;
      delete this.$t, delete this.$rt, delete this.$tc, delete this.$te, delete this.$d, delete this.$n, delete this.$tm, o.__disposer && (o.__disposer(), delete o.__disposer, delete o.__extender), n.__deleteInstance(r), delete this.$i18n;
    }
  };
}
function dl(e, t) {
  e.locale = t.locale || e.locale, e.fallbackLocale = t.fallbackLocale || e.fallbackLocale, e.missing = t.missing || e.missing, e.silentTranslationWarn = t.silentTranslationWarn || e.silentFallbackWarn, e.silentFallbackWarn = t.silentFallbackWarn || e.silentFallbackWarn, e.formatFallbackMessages = t.formatFallbackMessages || e.formatFallbackMessages, e.postTranslation = t.postTranslation || e.postTranslation, e.warnHtmlInMessage = t.warnHtmlInMessage || e.warnHtmlInMessage, e.escapeParameterHtml = t.escapeParameterHtml || e.escapeParameterHtml, e.sync = t.sync || e.sync, e.__composer[Dc](t.pluralizationRules || e.pluralizationRules);
  const n = xo(e.locale, {
    messages: t.messages,
    __i18n: t.__i18n
  });
  return Object.keys(n).forEach((r) => e.mergeLocaleMessage(r, n[r])), t.datetimeFormats && Object.keys(t.datetimeFormats).forEach((r) => e.mergeDateTimeFormat(r, t.datetimeFormats[r])), t.numberFormats && Object.keys(t.numberFormats).forEach((r) => e.mergeNumberFormat(r, t.numberFormats[r])), e;
}
const xb = /* @__PURE__ */ dn("global-vue-i18n");
function Db(e = {}, t) {
  const n = __VUE_I18N_LEGACY_API__ && ee(e.legacy) ? e.legacy : __VUE_I18N_LEGACY_API__, r = ee(e.globalInjection) ? e.globalInjection : !0, o = __VUE_I18N_LEGACY_API__ && n ? !!e.allowComposition : !0, i = /* @__PURE__ */ new Map(), [s, a] = Rb(e, n), l = /* @__PURE__ */ dn("");
  function u(f) {
    return i.get(f) || null;
  }
  function c(f, h) {
    i.set(f, h);
  }
  function d(f) {
    i.delete(f);
  }
  {
    const f = {
      // mode
      get mode() {
        return __VUE_I18N_LEGACY_API__ && n ? "legacy" : "composition";
      },
      // allowComposition
      get allowComposition() {
        return o;
      },
      // install plugin
      async install(h, ..._) {
        if (h.__VUE_I18N_SYMBOL__ = l, h.provide(h.__VUE_I18N_SYMBOL__, f), J(_[0])) {
          const P = _[0];
          f.__composerExtend = P.__composerExtend, f.__vueI18nExtend = P.__vueI18nExtend;
        }
        let E = null;
        !n && r && (E = Kb(h, f.global)), __VUE_I18N_FULL_INSTALL__ && Ib(h, f, ..._), __VUE_I18N_LEGACY_API__ && n && h.mixin(Ab(a, a.__composer, f));
        const w = h.unmount;
        h.unmount = () => {
          E && E(), f.dispose(), w();
        };
      },
      // global accessor
      get global() {
        return a;
      },
      dispose() {
        s.stop();
      },
      // @internal
      __instances: i,
      // @internal
      __getInstance: u,
      // @internal
      __setInstance: c,
      // @internal
      __deleteInstance: d
    };
    return f;
  }
}
function us(e = {}) {
  const t = jt();
  if (t == null)
    throw Ne($e.MUST_BE_CALL_SETUP_TOP);
  if (!t.isCE && t.appContext.app != null && !t.appContext.app.__VUE_I18N_SYMBOL__)
    throw Ne($e.NOT_INSTALLED);
  const n = Mb(t), r = jb(n), o = Mc(t), i = Fb(e, o);
  if (__VUE_I18N_LEGACY_API__ && n.mode === "legacy" && !e.__useComponent) {
    if (!n.allowComposition)
      throw Ne($e.NOT_AVAILABLE_IN_LEGACY_MODE);
    return Wb(t, i, r, e);
  }
  if (i === "global")
    return Fc(r, e, o), r;
  if (i === "parent") {
    let l = Ub(n, t, e.__useComponent);
    return l == null && (l = r), l;
  }
  const s = n;
  let a = s.__getInstance(t);
  if (a == null) {
    const l = Fe({}, e);
    "__i18n" in o && (l.__i18n = o.__i18n), r && (l.__root = r), a = as(l), s.__composerExtend && (a[Pi] = s.__composerExtend(a)), Hb(s, t, a), s.__setInstance(t, a);
  }
  return a;
}
function Rb(e, t, n) {
  const r = Sl();
  {
    const o = __VUE_I18N_LEGACY_API__ && t ? r.run(() => wi(e)) : r.run(() => as(e));
    if (o == null)
      throw Ne($e.UNEXPECTED_ERROR);
    return [r, o];
  }
}
function Mb(e) {
  {
    const t = cr(e.isCE ? xb : e.appContext.app.__VUE_I18N_SYMBOL__);
    if (!t)
      throw Ne(e.isCE ? $e.NOT_INSTALLED_WITH_PROVIDE : $e.UNEXPECTED_ERROR);
    return t;
  }
}
function Fb(e, t) {
  return No(e) ? "__i18n" in t ? "local" : "global" : e.useScope ? e.useScope : "local";
}
function jb(e) {
  return e.mode === "composition" ? e.global : e.global.__composer;
}
function Ub(e, t, n = !1) {
  let r = null;
  const o = t.root;
  let i = Vb(t, n);
  for (; i != null; ) {
    const s = e;
    if (e.mode === "composition")
      r = s.__getInstance(i);
    else if (__VUE_I18N_LEGACY_API__) {
      const a = s.__getInstance(i);
      a != null && (r = a.__composer, n && r && !r[Rc] && (r = null));
    }
    if (r != null || o === i)
      break;
    i = i.parent;
  }
  return r;
}
function Vb(e, t = !1) {
  return e == null ? null : t && e.vnode.ctx || e.parent;
}
function Hb(e, t, n) {
  Vr(() => {
  }, t), Ki(() => {
    const r = n;
    e.__deleteInstance(t);
    const o = r[Pi];
    o && (o(), delete r[Pi]);
  }, t);
}
function Wb(e, t, n, r = {}) {
  const o = t === "local", i = Ml(null);
  if (o && e.proxy && !(e.proxy.$options.i18n || e.proxy.$options.__i18n))
    throw Ne($e.MUST_DEFINE_I18N_OPTION_IN_ALLOW_COMPOSITION);
  const s = ee(r.inheritLocale) ? r.inheritLocale : !W(r.locale), a = De(
    // prettier-ignore
    !o || s ? n.locale.value : W(r.locale) ? r.locale : Yn
  ), l = De(
    // prettier-ignore
    !o || s ? n.fallbackLocale.value : W(r.fallbackLocale) || Se(r.fallbackLocale) || J(r.fallbackLocale) || r.fallbackLocale === !1 ? r.fallbackLocale : a.value
  ), u = De(xo(a.value, r)), c = De(J(r.datetimeFormats) ? r.datetimeFormats : { [a.value]: {} }), d = De(J(r.numberFormats) ? r.numberFormats : { [a.value]: {} }), f = o ? n.missingWarn : ee(r.missingWarn) || cn(r.missingWarn) ? r.missingWarn : !0, h = o ? n.fallbackWarn : ee(r.fallbackWarn) || cn(r.fallbackWarn) ? r.fallbackWarn : !0, _ = o ? n.fallbackRoot : ee(r.fallbackRoot) ? r.fallbackRoot : !0, E = !!r.fallbackFormat, w = ve(r.missing) ? r.missing : null, P = ve(r.postTranslation) ? r.postTranslation : null, M = o ? n.warnHtmlMessage : ee(r.warnHtmlMessage) ? r.warnHtmlMessage : !0, S = !!r.escapeParameter, g = o ? n.modifiers : J(r.modifiers) ? r.modifiers : {}, O = r.pluralRules || o && n.pluralRules;
  function L() {
    return [
      a.value,
      l.value,
      u.value,
      c.value,
      d.value
    ];
  }
  const A = dt({
    get: () => i.value ? i.value.locale.value : a.value,
    set: (v) => {
      i.value && (i.value.locale.value = v), a.value = v;
    }
  }), F = dt({
    get: () => i.value ? i.value.fallbackLocale.value : l.value,
    set: (v) => {
      i.value && (i.value.fallbackLocale.value = v), l.value = v;
    }
  }), $ = dt(() => i.value ? i.value.messages.value : u.value), B = dt(() => c.value), Y = dt(() => d.value);
  function R() {
    return i.value ? i.value.getPostTranslationHandler() : P;
  }
  function z(v) {
    i.value && i.value.setPostTranslationHandler(v);
  }
  function ae() {
    return i.value ? i.value.getMissingHandler() : w;
  }
  function Te(v) {
    i.value && i.value.setMissingHandler(v);
  }
  function ne(v) {
    return L(), v();
  }
  function te(...v) {
    return i.value ? ne(() => Reflect.apply(i.value.t, null, [...v])) : ne(() => "");
  }
  function Q(...v) {
    return i.value ? Reflect.apply(i.value.rt, null, [...v]) : "";
  }
  function Pe(...v) {
    return i.value ? ne(() => Reflect.apply(i.value.d, null, [...v])) : ne(() => "");
  }
  function we(...v) {
    return i.value ? ne(() => Reflect.apply(i.value.n, null, [...v])) : ne(() => "");
  }
  function ue(v) {
    return i.value ? i.value.tm(v) : {};
  }
  function he(v, N) {
    return i.value ? i.value.te(v, N) : !1;
  }
  function it(v) {
    return i.value ? i.value.getLocaleMessage(v) : {};
  }
  function Be(v, N) {
    i.value && (i.value.setLocaleMessage(v, N), u.value[v] = N);
  }
  function Lt(v, N) {
    i.value && i.value.mergeLocaleMessage(v, N);
  }
  function Qe(v) {
    return i.value ? i.value.getDateTimeFormat(v) : {};
  }
  function mt(v, N) {
    i.value && (i.value.setDateTimeFormat(v, N), c.value[v] = N);
  }
  function ht(v, N) {
    i.value && i.value.mergeDateTimeFormat(v, N);
  }
  function Pt(v) {
    return i.value ? i.value.getNumberFormat(v) : {};
  }
  function et(v, N) {
    i.value && (i.value.setNumberFormat(v, N), d.value[v] = N);
  }
  function Qt(v, N) {
    i.value && i.value.mergeNumberFormat(v, N);
  }
  const b = {
    get id() {
      return i.value ? i.value.id : -1;
    },
    locale: A,
    fallbackLocale: F,
    messages: $,
    datetimeFormats: B,
    numberFormats: Y,
    get inheritLocale() {
      return i.value ? i.value.inheritLocale : s;
    },
    set inheritLocale(v) {
      i.value && (i.value.inheritLocale = v);
    },
    get availableLocales() {
      return i.value ? i.value.availableLocales : Object.keys(u.value);
    },
    get modifiers() {
      return i.value ? i.value.modifiers : g;
    },
    get pluralRules() {
      return i.value ? i.value.pluralRules : O;
    },
    get isGlobal() {
      return i.value ? i.value.isGlobal : !1;
    },
    get missingWarn() {
      return i.value ? i.value.missingWarn : f;
    },
    set missingWarn(v) {
      i.value && (i.value.missingWarn = v);
    },
    get fallbackWarn() {
      return i.value ? i.value.fallbackWarn : h;
    },
    set fallbackWarn(v) {
      i.value && (i.value.missingWarn = v);
    },
    get fallbackRoot() {
      return i.value ? i.value.fallbackRoot : _;
    },
    set fallbackRoot(v) {
      i.value && (i.value.fallbackRoot = v);
    },
    get fallbackFormat() {
      return i.value ? i.value.fallbackFormat : E;
    },
    set fallbackFormat(v) {
      i.value && (i.value.fallbackFormat = v);
    },
    get warnHtmlMessage() {
      return i.value ? i.value.warnHtmlMessage : M;
    },
    set warnHtmlMessage(v) {
      i.value && (i.value.warnHtmlMessage = v);
    },
    get escapeParameter() {
      return i.value ? i.value.escapeParameter : S;
    },
    set escapeParameter(v) {
      i.value && (i.value.escapeParameter = v);
    },
    t: te,
    getPostTranslationHandler: R,
    setPostTranslationHandler: z,
    getMissingHandler: ae,
    setMissingHandler: Te,
    rt: Q,
    d: Pe,
    n: we,
    tm: ue,
    te: he,
    getLocaleMessage: it,
    setLocaleMessage: Be,
    mergeLocaleMessage: Lt,
    getDateTimeFormat: Qe,
    setDateTimeFormat: mt,
    mergeDateTimeFormat: ht,
    getNumberFormat: Pt,
    setNumberFormat: et,
    mergeNumberFormat: Qt
  };
  function y(v) {
    v.locale.value = a.value, v.fallbackLocale.value = l.value, Object.keys(u.value).forEach((N) => {
      v.mergeLocaleMessage(N, u.value[N]);
    }), Object.keys(c.value).forEach((N) => {
      v.mergeDateTimeFormat(N, c.value[N]);
    }), Object.keys(d.value).forEach((N) => {
      v.mergeNumberFormat(N, d.value[N]);
    }), v.escapeParameter = S, v.fallbackFormat = E, v.fallbackRoot = _, v.fallbackWarn = h, v.missingWarn = f, v.warnHtmlMessage = M;
  }
  return iu(() => {
    if (e.proxy == null || e.proxy.$i18n == null)
      throw Ne($e.NOT_AVAILABLE_COMPOSITION_IN_LEGACY);
    const v = i.value = e.proxy.$i18n.__composer;
    t === "global" ? (a.value = v.locale.value, l.value = v.fallbackLocale.value, u.value = v.messages.value, c.value = v.datetimeFormats.value, d.value = v.numberFormats.value) : o && y(v);
  }), b;
}
const Bb = [
  "locale",
  "fallbackLocale",
  "availableLocales"
], fl = ["t", "rt", "d", "n", "tm", "te"];
function Kb(e, t) {
  const n = /* @__PURE__ */ Object.create(null);
  return Bb.forEach((o) => {
    const i = Object.getOwnPropertyDescriptor(t, o);
    if (!i)
      throw Ne($e.UNEXPECTED_ERROR);
    const s = Me(i.value) ? {
      get() {
        return i.value.value;
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      set(a) {
        i.value.value = a;
      }
    } : {
      get() {
        return i.get && i.get();
      }
    };
    Object.defineProperty(n, o, s);
  }), e.config.globalProperties.$i18n = n, fl.forEach((o) => {
    const i = Object.getOwnPropertyDescriptor(t, o);
    if (!i || !i.value)
      throw Ne($e.UNEXPECTED_ERROR);
    Object.defineProperty(e.config.globalProperties, `$${o}`, i);
  }), () => {
    delete e.config.globalProperties.$i18n, fl.forEach((o) => {
      delete e.config.globalProperties[`$${o}`];
    });
  };
}
_b();
__INTLIFY_JIT_COMPILATION__ ? Ka(mb) : Ka(pb);
ob(Fg);
ib(Ec);
if (__INTLIFY_PROD_DEVTOOLS__) {
  const e = Gt();
  e.__INTLIFY__ = !0, zg(e.__INTLIFY_DEVTOOLS_GLOBAL_HOOK__);
}
const Yb = {
  en: {
    opencut: {
      title: "OpenCut"
    }
  },
  zh: {
    opencut: {
      title: "OpenCut"
    }
  }
}, zb = Db({
  legacy: !1,
  locale: navigator.language.split("-")[0] || "en",
  fallbackLocale: "en",
  messages: Yb
}), { ComfyButton: Gb } = window.comfyAPI.button;
let Qn = null, er = null, to = null;
function Xb() {
  return er && to || (er = document.createElement("div"), er.id = "opencut-root", document.body.appendChild(er), Qn = lp(Mh), Qn.use(fp()), Qn.use(zb), Qn.use(gm, {
    theme: "none"
  }), to = Qn.mount(er)), to;
}
function Jb() {
  Xb().open();
}
fs.registerExtension({
  name: "ComfyUI.OpenCut.TopMenu",
  setup() {
    var e;
    (e = fs.menu) == null || e.settingsGroup.append(
      new Gb({
        icon: "video",
        tooltip: "comfyui-opencut",
        content: "OpenCut",
        action: Jb
      })
    );
  }
});
