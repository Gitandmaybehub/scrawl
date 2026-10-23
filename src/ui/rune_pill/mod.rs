//! `scrawl lab pill <scene> --rune`: the capsule drawn by Rune instead of the Metal panel, on the
//! lab's timeline and pretend recording. It runs Rune's event loop (not GPUI's) in a transparent
//! borderless window placed where the capsule's panel goes, so moving it into an overlay later
//! changes only placement.
//!
//! What it shows and where everything rests is decided in `surface` (one `Surface`, Rune's
//! state); the view below only draws it, and Rune's springs animate between commits. The red
//! dot and the spinner animate through `Live` values a pretend meter thread publishes, without
//! re-running the view.

mod surface;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use rune::color::srgb_decode;
use rune::{
    App, AppSpec, Color, Compositor, Cx, El, Frame, LineCap, Live, Path, RunOptions, Shape, SpringPair,
    TextStyle, Transition, Tx, WindowOptions, WindowSpec, canvas, el, row, text,
};

use self::surface::{DONE_TEXT_X, LEAD_TEXT_X, PANEL_H, PROCESSING_TEXT_X, Shown, Surface};
use super::lab::{BACKDROP, Scene, Step, Timeline};
use super::{SelectionMode, motion};
use crate::native::pill::model::{CHIP_H, ElemId, HUD_TEXT, PANEL_W, PANEL_Y, Pres};

pub struct Options {
    pub scene: Scene,
    pub repeat: u32,
    pub backdrop: bool,
    /// Headless goldens: a `.png` renders `state`; a directory renders every state into it.
    pub snap: Option<PathBuf>,
    pub state: Option<String>,
}

/// The states `--snap` renders, as they rest.
const STATES: &[&str] = &[
    "picker",
    "picker-area",
    "recording-chips",
    "recording",
    "processing",
    "completed",
    "completed-routes",
];

fn state_pres(name: &str) -> Option<(Pres, &'static str)> {
    let pres = match name {
        "picker" => Pres::Selecting { mode: SelectionMode::Screen, has_iphone: true },
        "picker-area" => Pres::Selecting { mode: SelectionMode::Area, has_iphone: true },
        "recording-chips" => Pres::Recording { chips: true, has_iphone: true },
        "recording" => Pres::Recording { chips: false, has_iphone: true },
        "processing" => Pres::Processing,
        "completed" => Pres::Completed { label: "Copied".into(), routes: Vec::new() },
        "completed-routes" => Pres::Completed {
            label: "Copied".into(),
            routes: vec!["Send to scrawl".into(), "Send to rune".into()],
        },
        _ => return None,
    };
    Some((pres, "0:07"))
}

// ─── Colours (SwiftUI `PillColor`) ───────────────────────────────────────────────────────────

fn srgb(r: f32, g: f32, b: f32, a: f32) -> Color {
    Color { r: srgb_decode(r), g: srgb_decode(g), b: srgb_decode(b), a }
}

fn white(level: f32) -> Color {
    srgb(level, level, level, 1.0)
}

const SHELL: [f32; 3] = [20.0 / 255.0, 20.0 / 255.0, 22.0 / 255.0];

/// White at `alpha` over the capsule, as AppKit composites it (in sRGB): Rune blends in linear
/// light, where the same alpha reads far brighter.
fn veil(alpha: f32) -> Color {
    rune::theme::srgb_equivalent(Color::WHITE.alpha(alpha), srgb(SHELL[0], SHELL[1], SHELL[2], 1.0))
}

fn text_color() -> Color {
    srgb(0.961, 0.961, 0.969, 1.0)
}

fn red() -> Color {
    srgb(0.941, 0.267, 0.220, 1.0)
}

fn green() -> Color {
    srgb(0.412, 0.847, 0.624, 1.0)
}

fn accent(hover: bool) -> Color {
    if hover { srgb(0.294, 0.420, 1.0, 1.0) } else { srgb(0.231, 0.361, 0.965, 1.0) }
}

const MEDIUM: TextStyle = TextStyle::new(HUD_TEXT, 500.0);
const SEMIBOLD: TextStyle = TextStyle::new(HUD_TEXT, 600.0);

// ─── Text ────────────────────────────────────────────────────────────────────────────────────

static FONTS: std::sync::OnceLock<&'static [&'static [u8]]> = std::sync::OnceLock::new();

thread_local! {
    static TEXT: std::cell::RefCell<rune::text::TextSystem> = std::cell::RefCell::new({
        let fonts = FONTS.get().copied().unwrap_or(&[]);
        rune::text::TextSystem::new(fonts, families(fonts))
    });
}

fn families(fonts: &[&[u8]]) -> (&'static str, &'static str) {
    if fonts.is_empty() { ("Inter", "Inter") } else { ("System Font", "System Font") }
}

/// `text`'s width in points as Rune draws it (13 pt, medium or semibold), so chips hug the
/// labels they show.
pub fn measure(text: &str, semibold: bool) -> f32 {
    let style = if semibold { SEMIBOLD } else { MEDIUM };
    TEXT.with_borrow_mut(|t| t.measure(text, style, 2.0).0)
}

fn metrics() -> crate::native::pill::model::Metrics {
    crate::native::pill::model::Metrics {
        area: measure("Area", false),
        window: measure("Window", false),
        screen: measure("Screen", false),
        iphone: measure("iPhone", false),
        record: measure("Record", true),
        cancel: measure("Cancel", false),
        finish: measure("Finish", true),
        processing: measure("Processing…", true),
        paste: measure("Paste", true),
        inspect: measure("Inspect", false),
        clock_000: measure("0:00", true),
        clock_0000: measure("00:00", true),
    }
}

// ─── Live values ─────────────────────────────────────────────────────────────────────────────

/// What the pretend meter thread animates.
#[derive(Clone)]
struct Meter {
    /// The dot's core breath, its ripple ring's scale and fade (one 1.6 s cycle).
    core: Live<f32>,
    ring_scale: Live<f32>,
    ring_alpha: Live<f32>,
    /// The spinner's eight dots.
    spin: [Live<f32>; 8],
    /// 0 idle, 1 recording, 2 processing (written by the UI, read by the thread).
    activity: Arc<AtomicU8>,
    time_scale: Arc<AtomicU32>,
}

impl Meter {
    fn new() -> Self {
        let meter = Self {
            core: Live::new(1.0),
            ring_scale: Live::new(0.4),
            ring_alpha: Live::new(0.0),
            spin: std::array::from_fn(|_| Live::new(0.18)),
            activity: Arc::default(),
            time_scale: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
        };
        meter.publish(0.0);
        meter
    }

    /// The values at animation second `t` (the SwiftUI `pillDot` shader and `PillSpinner`).
    fn publish(&self, t: f64) {
        let phase = (t / 1.6).fract() as f32;
        let breath = 0.5 + 0.5 * (std::f32::consts::TAU * phase).cos();
        self.core.set(0.78 + 0.22 * breath);
        self.ring_scale.set((4.0 + 6.0 * phase) / 10.0);
        self.ring_alpha.set(0.34 * (1.0 - phase) * (1.0 - phase));
        let head = (t * 8.0).rem_euclid(8.0) as f32;
        for (k, dot) in self.spin.iter().enumerate() {
            let phase = (head - k as f32).rem_euclid(8.0);
            dot.set((1.0 - phase / 7.5).clamp(0.18, 1.0));
        }
    }

    /// Publishes at 120 Hz while something animates, the way an audio callback would.
    fn spawn(self) {
        std::thread::Builder::new()
            .name("rune-pill-meter".into())
            .spawn(move || {
                let mut t = 0.0_f64;
                let mut last = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_millis(8));
                    let now = Instant::now();
                    let scale = f32::from_bits(self.time_scale.load(Ordering::Relaxed));
                    t += (now - last).as_secs_f64() * f64::from(scale);
                    last = now;
                    if self.activity.load(Ordering::Relaxed) != 0 {
                        self.publish(t);
                    }
                }
            })
            .expect("meter thread");
    }
}

// ─── State ───────────────────────────────────────────────────────────────────────────────────

pub struct Lab {
    /// This window is the plain backdrop behind the capsule.
    backdrop: Option<Color>,
    /// A headless golden: the state rests, nothing plays.
    still: bool,
    surface: Surface,
    meter: Meter,
    /// The scene's steps, shared with the Metal lab.
    timeline: Timeline,
    recording_since: Option<f64>,
    now: f64,
}

impl Lab {
    fn capsule(scene: Scene, repeat: u32, meter: Meter) -> Self {
        motion::poll_tune();
        let metrics = metrics();
        Self {
            backdrop: None,
            still: false,
            surface: Surface::new(motion::tune(), metrics),
            meter,
            timeline: Timeline::new(scene, repeat),
            recording_since: None,
            now: 0.0,
        }
    }

    fn apply(&mut self, step: Step, now: f64) {
        motion::trace_event(format!("lab:{step:?}"));
        let anim = self.surface.anim(now);
        let s = &mut self.surface;
        match step {
            Step::Tap => s.set(Pres::Recording { chips: true, has_iphone: false }, now),
            Step::Chip(mode) | Step::Click(mode) => s.click(ElemId::from_mode(mode), now),
            Step::Open => s.set(Pres::Selecting { mode: SelectionMode::Screen, has_iphone: false }, now),
            Step::Cancel | Step::Dismiss => s.set(Pres::Hidden, now),
            Step::Finish => s.finish(now),
        }
        self.track_recording(anim);
    }

    /// A recording that starts (or is discarded) resets the clock.
    fn track_recording(&mut self, anim: f64) {
        match self.surface.pres {
            Pres::Recording { .. } => {
                self.recording_since.get_or_insert(anim);
            }
            _ => self.recording_since = None,
        }
        let activity = match self.surface.pres {
            Pres::Recording { .. } => 1,
            Pres::Processing => 2,
            _ => 0,
        };
        self.meter.activity.store(activity, Ordering::Relaxed);
    }
}

fn init() -> Lab {
    Lab::capsule(Scene::Picker, 1, Meter::new())
}

/// The lab's frame hook: tune reload, due surface steps, the scene's steps (the same `Timeline`
/// the Metal lab steps), the clock.
fn poll(s: &mut Lab, now: f64, tx: &mut Tx) -> bool {
    if s.backdrop.is_some() || s.still {
        return false;
    }
    s.now = now;
    motion::poll_tune();
    let tune = motion::tune();
    s.surface.tune = tune;
    s.surface.set_time_scale(now, tune.time_scale);
    s.meter.time_scale.store(s.surface.time_scale.to_bits(), Ordering::Relaxed);
    let scale = f64::from(s.surface.time_scale);
    let anim = s.surface.anim(now);

    let mut changed = s.surface.tick(now);
    let due = s.timeline.advance(anim);
    for step in due.steps {
        s.apply(step, now);
        changed = true;
    }
    if changed {
        s.track_recording(anim);
    }
    if due.done {
        motion::trace_close();
        tx.quit();
        return changed;
    }

    // The clock reads off the animation time, like the app's pretend recording.
    let mut wake = s.surface.next_due().map(|due| due - now);
    if let Some(since) = s.recording_since {
        let secs = (anim - since).max(0.0);
        let whole = secs as u64;
        let clock = format!("{}:{:02}", whole / 60, whole % 60);
        if clock != s.surface.clock {
            s.surface.set_clock(&clock, now);
            changed = true;
        }
        let tick = (whole as f64 + 1.0 - secs) / scale;
        wake = Some(wake.map_or(tick, |w| w.min(tick)));
    }
    if let Some(at) = s.timeline.next_at() {
        let step = (at - anim) / scale;
        wake = Some(wake.map_or(step, |w| w.min(step)));
    }
    if let Some(wake) = wake {
        tx.wake_after(Duration::from_secs_f64(wake.max(0.0)));
    }

    for event in s.surface.events.drain(..) {
        motion::trace_event(event);
    }
    if std::mem::take(&mut s.surface.instant) {
        tx.instant();
    }
    changed
}

// ─── View ────────────────────────────────────────────────────────────────────────────────────

fn view(s: &Lab, cx: &mut Cx<Lab>) -> El {
    if let Some(fill) = s.backdrop {
        return el().w_full().h_full().fill(fill);
    }
    let surface = &s.surface;
    cx.time_scale(surface.time_scale);
    cx.reduced_motion(surface.reduced);
    let mut root = el().size(PANEL_W, PANEL_H);
    if s.still {
        // Goldens show the capsule over the lab's backdrop, so its shadow and rim read.
        root = root.fill(srgb(0.851, 0.851, 0.871, 1.0));
    }
    root = root.child(shell(surface));
    let presence = Transition::blur(surface.tune.reveal_blur_px)
        .combined(Transition::scale(0.96))
        .springs(SpringPair::with_exit_ratio(presence_spring(surface), surface::EXIT_RATIO));
    if let Some(h) = &surface.highlight {
        root = root.child(
            placed(el(), h)
                .radius(CHIP_H * 0.5)
                .gradient(veil(0.216), veil(0.16))
                .transition(presence),
        );
    }
    // Content entering over content that leaves (lead → Processing → done, chips → Finish)
    // crossfades additively, without a midpoint alpha dip. The shell and the highlight are shared
    // backgrounds, so they stay outside; nothing inside overlaps at rest.
    let mut content = el().key("content").absolute(0.0, 0.0).size(PANEL_W, PANEL_H).dissolve();
    for shown in &surface.items {
        content = content.child(item(s, shown).transition(presence));
    }
    root.child(content)
}

fn presence_spring(surface: &Surface) -> rune::Spring {
    rune::Spring::new((surface.tune.reveal_enter_s * 1.6).clamp(0.05, 2.0), 0.0)
}

fn placed(e: El, shown: &Shown) -> El {
    let [x, y, w, h] = shown.rect;
    let e = e.key(shown.key.clone()).absolute(x, y).size(w, h);
    if shown.moved { e.motion(shown.motion) } else { e }
}

fn shell(surface: &Surface) -> El {
    let [x, y, w, h] = surface.shell;
    let completed = matches!(surface.pres, Pres::Completed { .. });
    let (rim_w, rim) = if completed { (0.75, green().alpha(0.5)) } else { (0.5, veil(0.09)) };
    el().key("shell")
        .absolute(x, y)
        .size(w, h)
        .radius(h * 0.5)
        // SwiftUI's 0.96, blended in sRGB there; in Rune's linear blend a light desktop would
        // lift it visibly, so nearly opaque.
        .fill(srgb(SHELL[0], SHELL[1], SHELL[2], 0.99))
        .border(rim_w, rim)
        .shadow(4.0, 18.0, Color::BLACK.alpha(0.34))
        .opacity(if surface.shell_on { 1.0 } else { 0.0 })
        .motion(surface.shell_motion)
        .opacity_motion(surface.shell_fade)
}

fn label(content: &str, style: TextStyle, color: Color) -> El {
    text(content).font(style).color(color)
}

fn item(s: &Lab, shown: &Shown) -> El {
    let id = shown.id;
    let press = move |lab: &mut Lab| {
        let now = lab.now;
        let anim = lab.surface.anim(now);
        match id {
            ElemId::Close | ElemId::Cancel | ElemId::Paste | ElemId::Inspect | ElemId::Route0 | ElemId::Route1 => {
                lab.surface.set(Pres::Hidden, now);
            }
            _ => lab.surface.click(id, now),
        }
        lab.track_recording(anim);
    };
    let base = placed(row().center(), shown);
    match id {
        ElemId::ChipArea | ElemId::ChipWindow | ElemId::ChipScreen | ElemId::ChipIphone => {
            let fg = white(if shown.active { 0.961 } else { 0.608 });
            let name = id.accessibility_label();
            let mut chip = base.radius(CHIP_H * 0.5).gap(5.0)
                .child(canvas(icon(id, 14.0 * 2.5 / 24.0, fg)).size(14.0, 14.0))
                .child(label(name, MEDIUM, fg))
                .press(veil(0.06));
            if !shown.active {
                chip = chip.hover(veil(0.08));
            }
            chip.on_press(press)
        }
        ElemId::Record | ElemId::Finish | ElemId::Paste => {
            let name = id.accessibility_label();
            base.radius(CHIP_H * 0.5)
                .fill(accent(false))
                .hover(accent(true))
                .child(label(name, SEMIBOLD, srgb(0.95, 0.95, 0.96, 1.0)))
                .on_press(press)
        }
        ElemId::Cancel | ElemId::Inspect | ElemId::Route0 | ElemId::Route1 => {
            let name = match (id, &s.surface.pres) {
                (ElemId::Route0, Pres::Completed { routes, .. }) => routes.first().cloned().unwrap_or_default(),
                (ElemId::Route1, Pres::Completed { routes, .. }) => routes.get(1).cloned().unwrap_or_default(),
                _ => id.accessibility_label().to_owned(),
            };
            base.radius(CHIP_H * 0.5)
                .hover(veil(0.08))
                .press(veil(0.06))
                .child(label(&name, MEDIUM, white(0.74)))
                .on_press(press)
        }
        ElemId::Close => base.child(
            row().center().size(28.0, 28.0).radius(14.0).hover(veil(0.10)).press(veil(0.06))
                .child(canvas(icon(id, 12.0 * 2.4 / 24.0, white(0.61))).size(12.0, 12.0))
                .on_press(press),
        ),
        ElemId::Lead => placed(el(), shown)
            .child(dot(&s.meter))
            .child(
                row().center_y().absolute(LEAD_TEXT_X, 0.0).h(CHIP_H)
                    .child(label(&s.surface.clock, SEMIBOLD, text_color())),
            ),
        ElemId::Processing => placed(el(), shown)
            .child(spinner(&s.meter))
            .child(
                row().center_y().absolute(PROCESSING_TEXT_X, 0.0).h(CHIP_H)
                    .child(label("Processing…", SEMIBOLD, text_color())),
            ),
        ElemId::DoneLead => {
            let done = match &s.surface.pres {
                Pres::Completed { label, .. } => label.clone(),
                _ => String::new(),
            };
            placed(el(), shown)
                .child(el().absolute(5.0, 5.0).size(22.0, 22.0).radius(11.0).fill(green().alpha(0.22)))
                .child(
                    canvas(icon(id, 13.0 * 2.6 / 24.0, green())).absolute(9.5, 9.5).size(13.0, 13.0),
                )
                .child(
                    row().center_y().absolute(DONE_TEXT_X, 0.0).h(CHIP_H)
                        .child(label(&done, SEMIBOLD, text_color())),
                )
        }
        ElemId::DividerLead | ElemId::DividerClose => placed(el(), shown),
    }
}

/// The recording dot: a breathing core with a glow, and a ripple ring every 1.6 s.
fn dot(meter: &Meter) -> El {
    let ring = 20.0;
    let core = 8.0;
    el().absolute(0.0, 0.0).size(CHIP_H, CHIP_H)
        .child(
            el().absolute((CHIP_H - ring) * 0.5, (CHIP_H - ring) * 0.5).size(ring, ring).radius(ring * 0.5)
                .fill(red().alpha(0.5))
                .live_scale(&meter.ring_scale)
                .live_opacity(&meter.ring_alpha),
        )
        .child(
            el().absolute((CHIP_H - core) * 0.5, (CHIP_H - core) * 0.5).size(core, core).radius(core * 0.5)
                .fill(red())
                .shadow(0.0, 7.0, red().alpha(0.55))
                .live_opacity(&meter.core),
        )
}

/// Eight dots around a 6.2 pt circle with a bright head going round.
fn spinner(meter: &Meter) -> El {
    let mut box_ = el().absolute(0.0, 0.0).size(CHIP_H, CHIP_H);
    let d = 2.7;
    for (k, live) in meter.spin.iter().enumerate() {
        let ang = k as f32 * std::f32::consts::FRAC_PI_4 - std::f32::consts::FRAC_PI_2;
        let (cx, cy) = (CHIP_H * 0.5 + 6.2 * ang.cos(), CHIP_H * 0.5 + 6.2 * ang.sin());
        box_ = box_.child(
            el().absolute(cx - d * 0.5, cy - d * 0.5).size(d, d).radius(d * 0.5).fill(text_color())
                .live_opacity(live),
        );
    }
    box_
}

/// A rounded rectangle on the icon grid.
fn rounded(x: f32, y: f32, w: f32, h: f32, r: f32) -> Path {
    let k = 0.552_284_8 * r;
    let (x1, y1) = (x + w, y + h);
    Path::new()
        .move_to([x + r, y])
        .line_to([x1 - r, y])
        .cubic_to([x1 - r + k, y], [x1, y + r - k], [x1, y + r])
        .line_to([x1, y1 - r])
        .cubic_to([x1, y1 - r + k], [x1 - r + k, y1], [x1 - r, y1])
        .line_to([x + r, y1])
        .cubic_to([x + r - k, y1], [x, y1 - r + k], [x, y1 - r])
        .line_to([x, y + r])
        .cubic_to([x, y + r - k], [x + r - k, y], [x + r, y])
        .close()
}

/// Scrawl's line icons on their 24 × 24 grid (`PillIcon`).
fn icon(id: ElemId, width: f32, color: Color) -> Shape {
    let mut paths: Vec<Path> = Vec::new();
    let line = |a: [f32; 2], b: [f32; 2]| Path::new().move_to(a).line_to(b);
    match id {
        ElemId::ChipArea => {
            for (a, c, b) in [
                ([3.0, 8.0], [3.0, 3.0], [8.0, 3.0]),
                ([16.0, 3.0], [21.0, 3.0], [21.0, 8.0]),
                ([21.0, 16.0], [21.0, 21.0], [16.0, 21.0]),
                ([8.0, 21.0], [3.0, 21.0], [3.0, 16.0]),
            ] {
                paths.push(Path::new().move_to(a).line_to(c).line_to(b));
            }
        }
        ElemId::ChipWindow => {
            paths.push(rounded(2.5, 4.0, 19.0, 16.0, 2.5));
            paths.push(line([2.5, 9.5], [21.5, 9.5]));
        }
        ElemId::ChipScreen => {
            paths.push(rounded(2.0, 3.0, 20.0, 14.0, 2.0));
            paths.push(line([8.0, 21.0], [16.0, 21.0]));
            paths.push(line([12.0, 17.0], [12.0, 21.0]));
        }
        ElemId::ChipIphone => {
            paths.push(rounded(5.5, 2.0, 13.0, 20.0, 3.0));
            paths.push(line([10.5, 6.0], [13.5, 6.0]));
        }
        ElemId::Close => {
            paths.push(line([6.5, 6.5], [17.5, 17.5]));
            paths.push(line([17.5, 6.5], [6.5, 17.5]));
        }
        ElemId::DoneLead => paths.push(Path::new().move_to([5.5, 12.5]).line_to([10.0, 17.0]).line_to([18.5, 7.5])),
        _ => {}
    }
    paths
        .iter()
        .fold(Shape::new().view_box(24.0, 24.0), |shape, path| shape.stroke_path(path, width, LineCap::Round, color))
}

// ─── Motion trace ────────────────────────────────────────────────────────────────────────────

/// Draws Rune's frame, then writes it to the motion-lint trace with the Metal capsule's names
/// (`shell`, `lead`, `chip:Area`, … in screen points), so the two traces compare.
struct TraceFrames {
    origin: [f32; 2],
}

impl Compositor<Lab> for TraceFrames {
    fn render(&mut self, f: &mut Frame<'_, Lab>) {
        f.draw_ui();
        if !motion::tracing() || f.rt.state.backdrop.is_some() {
            return;
        }
        let [ox, oy] = self.origin;
        let at = |r: [f32; 4]| [ox + r[0], oy + r[1], r[2], r[3]];
        let elements = f.rt.elements();
        motion::trace_frame();
        motion::trace_element("window", [ox, oy, PANEL_W, PANEL_H], 0.0, 0.0, None, None);
        let Some(shell) = elements.iter().find(|e| e.key.as_deref() == Some("shell")) else { return };
        let shell_rect = at(shell.rect);
        motion::trace_element("shell", shell_rect, shell.opacity, shell.rect[3] * 0.5, None, None);
        motion::trace_channel("content.blur", "level", 0.0);
        motion::trace_channel("content.alpha", "level", shell.opacity);
        // Entry lands its pose while invisible: a width jump there isn't motion anyone sees.
        if shell.opacity > 0.001 {
            motion::trace_channel("shell.width", "position", shell.rect[2]);
        }
        // Newest generation first, so an element and its exiting predecessor don't share a name.
        let mut items: Vec<_> = elements
            .iter()
            .filter_map(|e| {
                let (name, generation) = e.key.as_deref()?.split_once('#')?;
                (name != "highlight").then(|| (name, generation.parse::<u64>().unwrap_or(0), e))
            })
            .collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));
        let mut seen: Vec<&str> = Vec::new();
        for (name, _, e) in items {
            let key = if seen.contains(&name) { format!("{name}.old") } else { name.to_owned() };
            seen.push(name);
            motion::trace_element(key, at(e.rect), e.opacity, 0.0, Some("row"), Some(shell_rect));
        }
    }
}

// ─── Running ───────────────────────────────────────────────────────────────────────────────

/// What Rune's window options don't cover, set on the native window as the Metal panel sets
/// them: no window shadow (AppKit outlines a transparent window's opaque pixels with one), the
/// status-window level plus `level`, every Space, never hidden on deactivation.
fn panel_window(window: rune::WindowHandle, level: isize) {
    use objc2_app_kit::{NSStatusWindowLevel, NSView, NSWindowCollectionBehavior};
    use rune::raw_window_handle::RawWindowHandle;
    let result = window.with_raw_handle(|handle| {
        let RawWindowHandle::AppKit(appkit) = handle.as_raw() else { return };
        // SAFETY: on the main thread (`on_created`), the view lives as long as its window.
        let view: &NSView = unsafe { appkit.ns_view.cast().as_ref() };
        let Some(ns_window) = view.window() else { return };
        ns_window.setHasShadow(false);
        ns_window.setHidesOnDeactivate(false);
        ns_window.setLevel(NSStatusWindowLevel + level);
        ns_window.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::IgnoresCycle,
        );
    });
    if let Err(error) = result {
        eprintln!("rune lab: can't configure the window: {error}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────────

fn spec(size: (f32, f32), fonts: &'static [&'static [u8]]) -> AppSpec<Lab> {
    AppSpec {
        title: "Scrawl capsule (Rune)",
        size,
        min_size: Some((0.0, 0.0)),
        init,
        view,
        scenes: &[],
        fonts,
        families: families(fonts),
        setup: None,
        poll: Some(poll),
        on_key: None,
    }
}

/// SF (the capsule's face), read from the system at run time; Rune's Inter when it's missing.
fn system_fonts() -> &'static [&'static [u8]] {
    match std::fs::read("/System/Library/Fonts/SFNS.ttf") {
        Ok(bytes) => Box::leak(vec![&*Box::leak(bytes.into_boxed_slice())].into_boxed_slice()),
        Err(_) => &[],
    }
}

/// Rune reads its tooling flags from the process's arguments, so the lab re-execs itself with
/// arguments Rune reads the way clap did (`rune_args`) when they differ.
fn exec_with_rune_args() {
    use std::os::unix::process::CommandExt;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(args) = rune_args(&args) else { return };
    let Ok(exe) = std::env::current_exe() else { return };
    let error = std::process::Command::new(exe).args(args).exec();
    eprintln!("rune lab: couldn't re-exec for Rune's argument parser: {error}");
    std::process::exit(1);
}

/// `args` without Scrawl's `--trace` (a motion-lint file, already in `SCRAWL_MOTION_TRACE`),
/// which Rune would take for its own headless trace, and with `--snap=PATH` split into
/// `--snap PATH`, the only form Rune reads. `None` when nothing changes.
fn rune_args(args: &[String]) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(args.len());
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if arg == "--trace" {
            rest.next();
        } else if arg.starts_with("--trace=") {
            continue;
        } else if let Some(path) = arg.strip_prefix("--snap=") {
            out.extend(["--snap".to_owned(), path.to_owned()]);
        } else {
            out.push(arg.clone());
        }
    }
    (out != args).then_some(out)
}

/// `--snap DIR`: one headless run per state, each `--snap DIR/<state>.png --state <state>`.
fn snap_all(options: &Options, dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let exe = std::env::current_exe()?;
    for state in STATES {
        let out = dir.join(format!("{state}.png"));
        let status = std::process::Command::new(&exe)
            .args(["lab", "pill", options.scene.name(), "--rune", "--state", state, "--snap"])
            .arg(&out)
            .env_remove("SCRAWL_MOTION_TRACE")
            .status()?;
        anyhow::ensure!(status.success(), "snap {state} failed ({status})");
        println!("{}", out.display());
    }
    Ok(())
}

pub fn run(options: Options) -> anyhow::Result<()> {
    if let Some(snap) = &options.snap
        && snap.extension().is_none_or(|e| e != "png")
    {
        return snap_all(&options, snap);
    }
    exec_with_rune_args();
    let fonts = system_fonts();
    let _ = FONTS.set(fonts);
    let still = match options.snap.as_ref().map(|_| options.state.as_deref().unwrap_or("picker")) {
        Some(name) => Some(state_pres(name).ok_or_else(|| {
            anyhow::anyhow!("unknown state {name}; one of {}", STATES.join(", "))
        })?),
        None => None,
    };
    let Options { scene, repeat, backdrop, .. } = options;
    rune::run_windows_with(RunOptions { accessory: true }, move |displays| {
        // Where the Metal capsule's panel goes on the primary display, in global points.
        let screen = crate::screens::current().into_iter().next();
        let (frame, origin) = match &screen {
            Some(s) => {
                let [x, y] = crate::screens::pill_origin(s, f64::from(PANEL_W), f64::from(PANEL_Y));
                ([s.frame[2] as f32, s.frame[3] as f32], [x as f32, y as f32])
            }
            None => ([1440.0, 900.0], [400.0, PANEL_Y]),
        };
        let scale = displays.iter().find(|d| d.primary).or(displays.first()).map_or(2.0, |d| d.scale as f32);
        let physical = |x: f32, y: f32| ((x * scale).round() as i32, (y * scale).round() as i32);
        let place = |x: f32, y: f32| WindowOptions {
            borderless: true,
            always_on_top: true,
            position: Some(physical(x, y)),
            min_size: Some((0.0, 0.0)),
            resizable: false,
            ..WindowOptions::default()
        };

        let meter = Meter::new();
        let mut lab = Lab::capsule(scene, repeat, meter.clone());
        let mut windows = Vec::new();
        if let Some((pres, clock)) = still {
            lab.still = true;
            lab.surface.clock = clock.to_owned();
            let surface = std::mem::replace(&mut lab.surface, Surface::new(motion::tune(), Default::default()));
            lab.surface = surface.settled(pres);
        } else {
            motion::trace_open(&format!("pill.{}", scene.name()), frame);
            meter.clone().spawn();
            let backdrop_x = ((frame[0] - BACKDROP[2]) * 0.5).round();
            // For screen recordings (`perf/pill/record.sh`): the backdrop's rect in screen points.
            println!("region {},{},{},{}", backdrop_x, BACKDROP[1], BACKDROP[2], BACKDROP[3]);
            if backdrop {
                let theme = std::env::var("SCRAWL_LAB_BACKDROP").unwrap_or_default().to_ascii_lowercase();
                let fill = match theme.trim() {
                    "dark" | "busy-dark" => srgb(0.059, 0.067, 0.090, 1.0),
                    "light" | "white" => srgb(0.980, 0.980, 0.988, 1.0),
                    _ => srgb(0.851, 0.851, 0.871, 1.0),
                };
                let mut state = Lab::capsule(scene, repeat, meter);
                state.backdrop = Some(fill);
                windows.push(
                    WindowSpec::new("backdrop", spec((BACKDROP[2], BACKDROP[3]), fonts))
                        .state(state)
                        .options(place(backdrop_x, BACKDROP[1]))
                        .on_created(|window| panel_window(window, 0)),
                );
            }
        }
        // Opened after (so above) the backdrop.
        let app = App::new(spec((PANEL_W, PANEL_H), fonts)).compositor(TraceFrames { origin });
        windows.push(
            WindowSpec::from_app("capsule", app)
                .state(lab)
                .options(WindowOptions { transparent: true, ..place(origin[0], origin[1]) })
                .on_created(|window| panel_window(window, 5)),
        );
        windows
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::rune_args;

    fn args(s: &str) -> Vec<String> {
        s.split(' ').map(str::to_owned).collect()
    }

    #[test]
    fn rune_gets_snap_as_two_tokens_and_no_trace() {
        assert_eq!(
            rune_args(&args("lab pill finish --rune --snap=out.png --trace t.jsonl")),
            Some(args("lab pill finish --rune --snap out.png"))
        );
        assert_eq!(rune_args(&args("lab pill finish --trace=t.jsonl --rune")), Some(args("lab pill finish --rune")));
        assert_eq!(rune_args(&args("lab pill finish --rune --snap out.png")), None);
    }
}
