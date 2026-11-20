import { execFile } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { LanguageClient, LanguageClientOptions, ServerOptions } from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let output: vscode.OutputChannel;
let running = ""; // what the current server was started with, so a no-op change doesn't restart it

const WIN = process.platform === "win32";

function folder(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

/** A project's own virtualenv, as `uv sync`, `poetry` (in-project) or `python -m venv .venv` make it. */
function projectVenvPython(): string | undefined {
  const root = folder();
  if (!root) {
    return undefined;
  }
  for (const dir of [".venv", "venv"]) {
    const py = path.join(root, dir, WIN ? "Scripts" : "bin", WIN ? "python.exe" : "python");
    if (fs.existsSync(py)) {
      return py;
    }
  }
  return undefined;
}

/** The project's interpreter: `tensi.pythonPath`, else the Python extension's choice, else `.venv`. */
async function interpreter(): Promise<string | undefined> {
  const configured = vscode.workspace.getConfiguration("tensi").get<string>("pythonPath", "").trim();
  if (configured) {
    return configured;
  }
  const ext = vscode.extensions.getExtension("ms-python.python");
  if (ext) {
    try {
      const api = await ext.activate();
      const envs = api?.environments;
      const active = envs?.getActiveEnvironmentPath(vscode.workspace.workspaceFolders?.[0]?.uri);
      const env = active ? await envs.resolveEnvironment(active) : undefined;
      const py = env?.executable?.uri?.fsPath ?? active?.path;
      if (py && path.isAbsolute(py)) {
        return py;
      }
    } catch (err) {
      output.appendLine(`Could not ask the Python extension for an interpreter: ${err}`);
    }
  }
  return projectVenvPython();
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
 * Environment for the server. The project's interpreter goes first on PATH and in VIRTUAL_ENV, so
 * Pyrefly resolves torch and the project's own packages there, even when tensi itself is a global
 * tool (`uv tool install tensi-shapes`).
 */
function envFor(python: string | undefined): NodeJS.ProcessEnv {
  const env = { ...process.env };
  if (python && path.isAbsolute(python)) {
    const bin = path.dirname(python);
    env.PATH = bin + path.delimiter + (env.PATH ?? "");
    env.VIRTUAL_ENV = path.dirname(bin);
  }
  return env;
}

function run(cmd: string, args: string[], env: NodeJS.ProcessEnv, cwd?: string): Promise<{ ok: boolean; out: string }> {
  return new Promise((resolve) => {
    execFile(cmd, args, { env, cwd, timeout: 600_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      resolve({ ok: !err, out: `${stdout}${stderr}`.trim() });
    });
  });
}

/**
 * How to run tensi for this project: from the project's own environment if tensi is installed there,
 * else a global `tensi` (uv tool, pipx). Undefined when neither exists.
 */
async function tensiCommand(python: string | undefined): Promise<string[] | undefined> {
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
    output.appendLine(`${cmd.join(" ")} --version: ${probe.out || "(no output)"}`);
    if (probe.ok) {
      return cmd;
    }
  }
  return undefined;
}

async function offerInstall(python: string | undefined): Promise<boolean> {
  const uv = which("uv");
  const root = folder();
  const uvProject = !!(uv && root && fs.existsSync(path.join(root, "uv.lock")));
  const here = "Add to this project";
  const everywhere = "Install for all projects";
  const choose = "Choose interpreter";
  const actions = [...(python ? [here] : []), ...(uv ? [everywhere] : []), choose];
  const pick = await vscode.window.showWarningMessage(
    python
      ? `tensi isn't installed for this project (${python}).`
      : "tensi found no Python environment for this project. Pick an interpreter, or create one with `uv sync`.",
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
  let cmd: string[];
  if (pick === here && python) {
    if (uvProject && uv) {
      cmd = [uv, "add", "--dev", "tensi-shapes"]; // recorded in pyproject.toml, so `uv sync` keeps it
    } else if (uv) {
      cmd = [uv, "pip", "install", "--python", python, "tensi-shapes"]; // uv venvs have no pip
    } else {
      cmd = [python, "-m", "pip", "install", "--upgrade", "tensi-shapes"];
    }
  } else if (pick === everywhere && uv) {
    cmd = [uv, "tool", "install", "--upgrade", "tensi-shapes"];
  } else {
    return false;
  }
  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Installing tensi…" },
    async () => {
      output.appendLine(`$ ${cmd.join(" ")}`);
      const res = await run(cmd[0], cmd.slice(1), envFor(python), root);
      output.appendLine(res.out);
      if (!res.ok) {
        output.show(true);
        vscode.window.showErrorMessage("Installing tensi failed; see the tensi output for details.");
      }
      return res.ok;
    },
  );
}

async function start(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration("tensi");
  const python = await interpreter();
  let cmd = await tensiCommand(python);
  if (!cmd && (await offerInstall(python))) {
    cmd = await tensiCommand(python);
  }
  if (!cmd) {
    output.appendLine("tensi is not running: it isn't installed for this project or globally.");
    running = "";
    return;
  }
  output.appendLine(`Starting tensi: ${cmd.join(" ")} lsp  (project interpreter: ${python ?? "none"})`);
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
  await client.start();
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
      vscode.window.showErrorMessage(`tensi could not start: ${err}. Run "tensi: Check setup" for details.`);
    }
  });
  return restarting;
}

/** Restart only if the project's interpreter or tensi's location actually changed. */
async function refresh(): Promise<void> {
  const python = await interpreter();
  const cmd = await tensiCommand(python);
  if (JSON.stringify([cmd, python]) !== running) {
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
  const cmd = await tensiCommand(python);
  output.show(true);
  if (!cmd) {
    if (await offerInstall(python)) {
      await restart();
    }
    return;
  }
  const root = folder() ?? ".";
  const res = await run(cmd[0], [...cmd.slice(1), "doctor", root], { ...envFor(python), NO_COLOR: "1" }, root);
  output.appendLine(res.out);
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  output = vscode.window.createOutputChannel("tensi");
  const changed = debounce(() => void refresh(), 1500);
  // `uv sync`, `uv add`, a new `.venv`: pick up the environment without a manual restart
  const watcher = vscode.workspace.createFileSystemWatcher(
    "**/{uv.lock,poetry.lock,pyproject.toml,.venv/pyvenv.cfg,venv/pyvenv.cfg}",
  );
  context.subscriptions.push(
    output,
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
