import { execFile } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let output: vscode.OutputChannel;
let status: vscode.StatusBarItem;
let running = ""; // what the current server was started with, so a no-op change doesn't restart it
const warned = new Set<string>(); // one warning per problem per session
let extensionVersion = "0.0.0";
const versions = new Map<string, string>(); // tensi command -> the version it reported

const WIN = process.platform === "win32";
const BIN = WIN ? "Scripts" : "bin";
const PY = WIN ? "python.exe" : "python";

function folder(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

function inside(root: string | undefined, p: string): boolean {
  if (!root) {
    return false;
  }
  const rel = path.relative(root, p);
  return !!rel && !rel.startsWith("..") && !path.isAbsolute(rel);
}

function warnOnce(key: string, message: string): void {
  output.appendLine(message);
  if (!warned.has(key)) {
    warned.add(key);
    void vscode.window.showWarningMessage(message);
  }
}

/** A project's own virtualenv, as `uv sync`, `poetry` (in-project) or `python -m venv .venv` make it. */
function projectVenvPython(): string | undefined {
  const root = folder();
  if (!root) {
    return undefined;
  }
  for (const dir of [".venv", "venv", "env"]) {
    const py = path.join(root, dir, BIN, PY);
    if (fs.existsSync(py) && fs.existsSync(path.join(root, dir, "pyvenv.cfg"))) {
      return py;
    }
  }
  return undefined;
}

/** The Python extension's interpreter for this folder, if it has one installed and picked. */
async function pythonExtensionChoice(): Promise<string | undefined> {
  const ext = vscode.extensions.getExtension("ms-python.python");
  if (!ext) {
    return undefined;
  }
  try {
    const api = await ext.activate();
    const envs = api?.environments;
    const active = envs?.getActiveEnvironmentPath(vscode.workspace.workspaceFolders?.[0]?.uri);
    const env = active ? await envs.resolveEnvironment(active) : undefined;
    const py = env?.executable?.uri?.fsPath ?? active?.path;
    return py && path.isAbsolute(py) && fs.existsSync(py) ? py : undefined;
  } catch (err) {
    output.appendLine(`Could not ask the Python extension for an interpreter: ${err}`);
    return undefined;
  }
}

/**
 * The project's interpreter, which decides where torch and the project's packages come from:
 * 1. `tensi.pythonPath`, if it is set and exists (a stale path is reported and skipped);
 * 2. the project's own `.venv` / `venv`, what `uv sync` or `python -m venv` made for this folder;
 * 3. the Python extension's pick (a global Homebrew or system Python only when there is no venv).
 */
async function interpreter(): Promise<string | undefined> {
  const configured = vscode.workspace.getConfiguration("tensi").get<string>("pythonPath", "").trim();
  if (configured) {
    if (fs.existsSync(configured)) {
      return configured;
    }
    warnOnce(
      `pythonPath:${configured}`,
      `tensi: the "tensi.pythonPath" setting points to ${configured}, which doesn't exist. ` +
        "tensi ignores it and uses the project's environment; clear the setting to stop this warning.",
    );
  }
  const venv = projectVenvPython();
  const picked = await pythonExtensionChoice();
  if (picked && (inside(folder(), picked) || !venv)) {
    return picked;
  }
  return venv ?? picked;
}

/** An executable on PATH or in the usual per-user tool folders (a GUI-launched editor often lacks them). */
function which(name: string): string | undefined {
  const exe = WIN ? `${name}.exe` : name;
  const dirs = (process.env.PATH ?? "").split(path.delimiter);
  dirs.push(path.join(os.homedir(), ".local", "bin"), path.join(os.homedir(), ".cargo", "bin"));
  if (!WIN) {
    dirs.push("/opt/homebrew/bin", "/usr/local/bin");
  }
  for (const d of dirs) {
    const p = path.join(d, exe);
    if (d && fs.existsSync(p)) {
      return p;
    }
  }
  return undefined;
}

/**
 * Environment for the server. The project's interpreter goes first on PATH, and in VIRTUAL_ENV when
 * it lives in a virtualenv, so Pyrefly resolves torch and the project's own packages there, even
 * when tensi itself is a global tool (`uv tool install tensi-shapes`).
 */
function envFor(python: string | undefined): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (python && path.isAbsolute(python)) {
    const bin = path.dirname(python);
    env.PATH = bin + path.delimiter + (env.PATH ?? "");
    const prefix = path.dirname(bin);
    if (fs.existsSync(path.join(prefix, "pyvenv.cfg"))) {
      env.VIRTUAL_ENV = prefix;
    } else {
      delete env.VIRTUAL_ENV; // a system Python: let Pyrefly find the project's own
    }
  }
  return env;
}

function run(cmd: string, args: string[], env: NodeJS.ProcessEnv, cwd?: string): Promise<{ ok: boolean; out: string }> {
  return new Promise((resolve) => {
    execFile(cmd, args, { env, cwd, timeout: 600_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      const out = `${stdout}${stderr}`.trim();
      resolve({ ok: !err, out: out || (err ? err.message : "") });
    });
  });
}

/**
 * How to run tensi for this project: from the project's own environment if tensi is installed there,
 * else a global `tensi` (uv tool, pipx). Undefined when neither exists.
 */
async function tensiCommand(python: string | undefined, verbose: boolean): Promise<string[] | undefined> {
  const env = envFor(python);
  const candidates: string[][] = [];
  if (python) {
    candidates.push([python, "-m", "tensi"]);
  }
  const global = which("tensi");
  if (global) {
    candidates.push([global]);
  }
  for (const cmd of candidates) {
    const probe = await run(cmd[0], [...cmd.slice(1), "--version"], env);
    if (verbose) {
      output.appendLine(`${cmd.join(" ")} --version: ${probe.ok ? probe.out : `not usable (${probe.out.split("\n").pop()})`}`);
    }
    if (probe.ok) {
      versions.set(cmd.join(" "), probe.out.trim().split(/\s+/).pop() ?? "");
      return cmd;
    }
  }
  return undefined;
}

/** The install command that puts tensi where this project's interpreter will find it. */
function installCommand(python: string | undefined, root: string | undefined): { cmd: string[]; where: string } | undefined {
  const uv = which("uv");
  if (python) {
    const inProject = inside(root, python);
    if (uv && root && inProject && fs.existsSync(path.join(root, "uv.lock"))) {
      // recorded in pyproject.toml as a dev dependency, so `uv sync` keeps it
      return { cmd: [uv, "add", "--dev", "tensi-shapes"], where: "this project (uv add --dev)" };
    }
    if (uv) {
      return { cmd: [uv, "pip", "install", "--python", python, "tensi-shapes"], where: "this project's environment" };
    }
    return { cmd: [python, "-m", "pip", "install", "tensi-shapes"], where: "this project's environment" };
  }
  if (uv) {
    return { cmd: [uv, "tool", "install", "--upgrade", "tensi-shapes"], where: "all projects (uv tool)" };
  }
  return undefined;
}

async function install(python: string | undefined, global: boolean): Promise<boolean> {
  const root = folder();
  const uv = which("uv");
  const plan = global
    ? uv
      ? { cmd: [uv, "tool", "install", "--upgrade", "tensi-shapes"], where: "all projects (uv tool)" }
      : undefined
    : installCommand(python, root);
  if (!plan) {
    output.show(true);
    output.appendLine("tensi: no way to install found. Install uv (https://docs.astral.sh/uv/) and reload the window.");
    void vscode.window.showErrorMessage("tensi needs uv or pip to install itself. See the tensi output.");
    return false;
  }
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Installing tensi for ${plan.where}…` },
    async () => {
      output.appendLine(`$ ${plan.cmd.join(" ")}`);
      const res = await run(plan.cmd[0], plan.cmd.slice(1), envFor(python), root);
      output.appendLine(res.out);
      if (!res.ok) {
        output.show(true);
        void vscode.window.showErrorMessage("Installing tensi failed; see the tensi output for the command and its error.");
      }
      return res.ok;
    },
  );
}

async function offerInstall(python: string | undefined): Promise<boolean> {
  const here = "Install tensi";
  const everywhere = "Install for all projects";
  const choose = "Choose interpreter";
  const actions = [...(python ? [here] : []), ...(which("uv") ? [everywhere] : []), choose];
  const pick = await vscode.window.showInformationMessage(
    python
      ? `tensi shows tensor shapes on hover. Install it for this project (${python})?`
      : "tensi found no Python environment for this project. Create one (for example with `uv sync`), or pick an interpreter.",
    ...actions,
  );
  if (pick === choose) {
    const hasPython = !!vscode.extensions.getExtension("ms-python.python");
    await vscode.commands.executeCommand(
      hasPython ? "python.setInterpreter" : "workbench.action.openSettings",
      ...(hasPython ? [] : ["tensi.pythonPath"]),
    );
    return false;
  }
  if (pick === here || pick === everywhere) {
    return install(python, pick === everywhere);
  }
  return false;
}

/** Whether version `a` is older than `b` (plain x.y.z). */
function older(a: string, b: string): boolean {
  const pa = a.split(".").map((n) => parseInt(n, 10) || 0);
  const pb = b.split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) {
      return (pa[i] ?? 0) < (pb[i] ?? 0);
    }
  }
  return false;
}

/** The command that upgrades the tensi that `cmd` runs, in place. */
function updateCommand(cmd: string[], python: string | undefined): string[] | undefined {
  const uv = which("uv");
  const root = folder();
  if (cmd.length === 1) {
    // a global tensi: pipx if pipx installed it, else uv tool
    let real = cmd[0];
    try {
      real = fs.realpathSync(cmd[0]);
    } catch {
      // keep the path as found
    }
    const pipx = which("pipx");
    if (pipx && real.includes("pipx")) {
      return [pipx, "upgrade", "tensi-shapes"];
    }
    return uv ? [uv, "tool", "install", "--upgrade", "tensi-shapes"] : undefined;
  }
  if (uv && root && python && inside(root, python) && fs.existsSync(path.join(root, "uv.lock"))) {
    return [uv, "add", "--dev", "--upgrade-package", "tensi-shapes", "tensi-shapes"];
  }
  if (uv && python) {
    return [uv, "pip", "install", "--upgrade", "--python", python, "tensi-shapes"];
  }
  return python ? [python, "-m", "pip", "install", "--upgrade", "tensi-shapes"] : undefined;
}

/** tensi older than this extension (a stale global install, say): offer to update it once per session. */
async function offerUpdate(cmd: string[], python: string | undefined): Promise<void> {
  const have = versions.get(cmd.join(" ")) ?? "";
  if (!have || !older(have, extensionVersion) || warned.has(`update ${cmd.join(" ")}`)) {
    return;
  }
  warned.add(`update ${cmd.join(" ")}`);
  const up = updateCommand(cmd, python);
  output.appendLine(`tensi ${have} is older than this extension (${extensionVersion}).`);
  if (!up) {
    return;
  }
  const where = cmd.length === 1 ? "installed for all projects" : "in this project";
  const pick = await vscode.window.showInformationMessage(
    `The tensi ${where} is ${have}; this extension expects ${extensionVersion}. Update it?`,
    "Update tensi",
  );
  if (pick !== "Update tensi") {
    return;
  }
  const ok = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Updating tensi…" },
    async () => {
      output.appendLine(`$ ${up.join(" ")}`);
      const res = await run(up[0], up.slice(1), envFor(python), folder());
      output.appendLine(res.out);
      return res.ok;
    },
  );
  if (ok) {
    running = "";
    void restart();
  } else {
    output.show(true);
    void vscode.window.showErrorMessage("Updating tensi failed; see the tensi output.");
  }
}

function setStatus(state: "starting" | "solving" | "ready" | "off", detail = "", toAnnotate = 0): void {
  const icon = { starting: "$(sync~spin)", solving: "$(sync~spin)", ready: "$(symbol-array)", off: "$(warning)" }[state];
  const pending = state === "ready" && toAnnotate > 0;
  status.text = pending ? `${icon} tensi: ${toAnnotate} to annotate` : `${icon} tensi`;
  status.tooltip = {
    starting: "tensi is starting",
    solving: `tensi is working out shapes${detail ? ` in ${detail}` : ""}; hovers fill in when it's done`,
    ready: pending
      ? `${toAnnotate} input${toAnnotate === 1 ? "" : "s"} in ${detail} need a shape annotation: their shape comes ` +
        "from outside the file, so tensors computed from them show none. Each one has a blue hint with the exact " +
        "line to write and a quick fix (Ctrl+. / Cmd+.) that writes it."
      : `tensi is ready${detail ? `: ${detail}` : ""}. Click to check the setup.`,
    off: `tensi is not running${detail ? `: ${detail}` : ""}. Click to check the setup.`,
  }[state];
  status.show();
}

async function start(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("tensi");
  setStatus("starting");
  const python = await interpreter();
  let cmd = await tensiCommand(python, true);
  if (!cmd && (await offerInstall(python))) {
    cmd = await tensiCommand(python, true);
  }
  if (!cmd) {
    output.appendLine("tensi is not running: it isn't installed for this project or globally.");
    setStatus("off", "not installed");
    running = "";
    return;
  }
  output.appendLine(`Starting tensi: ${cmd.join(" ")} lsp  (project interpreter: ${python ?? "none found"})`);
  running = JSON.stringify([cmd, python]);
  const serverOptions: ServerOptions = {
    command: cmd[0],
    args: [...cmd.slice(1), "lsp"],
    options: { env: envFor(python), cwd: folder() },
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "python" }],
    initializationOptions: {
      hover: cfg.get("hover"),
      diagnostics: cfg.get("diagnostics"),
      inlayHints: cfg.get("inlayHints"),
      shadowWorkspace: cfg.get("shadowWorkspace"),
    },
    diagnosticCollectionName: "tensi",
    outputChannel: output,
  };
  client = new LanguageClient("tensi", "tensi", serverOptions, clientOptions);
  client.onNotification("$/tensi/status", (p: { uri?: string; state?: string; toAnnotate?: number }) => {
    const name = p.uri ? path.basename(vscode.Uri.parse(p.uri).fsPath) : "";
    const active = vscode.window.activeTextEditor?.document.uri.toString();
    if (p.uri && active && p.uri !== active && p.state !== "solving") {
      return; // a background file: keep showing the one in front
    }
    setStatus(p.state === "solving" ? "solving" : "ready", name, p.toAnnotate ?? 0);
  });
  await client.start();
  setStatus("ready", python ? `using ${python}` : "");
  void offerUpdate(cmd, python);
}

let restarting: Promise<void> = Promise.resolve();

/** Restart the server. Calls are serialized, so a burst of changes can't start two servers. */
function restart(): Promise<void> {
  restarting = restarting.then(async () => {
    const old = client;
    client = undefined;
    await old?.stop();
    try {
      await start();
    } catch (err) {
      output.appendLine(`tensi could not start: ${err}`);
      setStatus("off", String(err));
      void vscode.window.showErrorMessage(`tensi could not start: ${err}. Run "tensi: Check setup" for details.`);
    }
  });
  return restarting;
}

/** Restart only if the project's interpreter or tensi's location actually changed. */
async function refresh(): Promise<void> {
  const python = await interpreter();
  const cmd = await tensiCommand(python, false);
  if (JSON.stringify([cmd, python]) !== running) {
    output.appendLine("The project's environment changed; restarting tensi.");
    await restart();
  }
}

function debounce(fn: () => void, ms: number): () => void {
  let t: NodeJS.Timeout | undefined;
  return () => {
    if (t) {
      clearTimeout(t);
    }
    t = setTimeout(fn, ms);
  };
}

async function doctor(): Promise<void> {
  const python = await interpreter();
  const cmd = await tensiCommand(python, true);
  output.show(true);
  if (!cmd) {
    if (await offerInstall(python)) {
      await restart();
    }
    return;
  }
  const root = folder() ?? ".";
  output.appendLine(`Project interpreter: ${python ?? "none found"}`);
  const res = await run(cmd[0], [...cmd.slice(1), "doctor", root], { ...envFor(python), NO_COLOR: "1" }, root);
  output.appendLine(res.out);
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  output = vscode.window.createOutputChannel("tensi");
  extensionVersion = String(context.extension.packageJSON.version ?? "0.0.0");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  status.command = "tensi.doctor";
  const changed = debounce(() => void refresh(), 1500);
  // `uv sync`, `uv add`, a new `.venv`: pick up the environment without a manual restart
  const watcher = vscode.workspace.createFileSystemWatcher(
    "**/{uv.lock,poetry.lock,pyproject.toml,.venv/pyvenv.cfg,venv/pyvenv.cfg,env/pyvenv.cfg}",
  );
  context.subscriptions.push(
    output,
    status,
    watcher,
    watcher.onDidCreate(changed),
    watcher.onDidChange(changed),
    watcher.onDidDelete(changed),
    vscode.commands.registerCommand("tensi.restart", restart),
    vscode.commands.registerCommand("tensi.doctor", doctor),
    vscode.workspace.onDidChangeWorkspaceFolders(() => void restart()),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("tensi")) {
        void restart();
      }
    }),
  );
  const py = vscode.extensions.getExtension("ms-python.python");
  if (py) {
    try {
      const api = await py.activate();
      const sub = api?.environments?.onDidChangeActiveEnvironmentPath?.(changed);
      if (sub) {
        context.subscriptions.push(sub);
      }
    } catch {
      // the Python extension is optional
    }
  }
  await restart();
}

export async function deactivate(): Promise<void> {
  await client?.stop();
}
