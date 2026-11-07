"""SCAPS-1D optimization helpers for the FTO/WO3/MASnIBr2/CBTS/Au device.

Loaded by the notebook with  %run -i ../src/scaps_tools.py  so that every
function sees the configuration variables of the notebook (paths, ranges, seeds).
The code is the same as in the working notebook used for the paper."""


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 4 — DeviceConfig dataclass + utilities
# ============================================================
# All device settings live in DeviceConfig. Text IO and regex/math helpers
# used by the .scaps builder below.
# ============================================================

@dataclass
class Layer:
    name: str
    block_name: str
    role: str
    doping_type: str  # 'n', 'p', 'both'
    thickness_nm: Optional[float]
    optimize: bool

@dataclass
class DeviceConfig:
    """Holds all per-device state. One instance per sub-cell (for tandems)."""
    layers: List[Layer] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    interface_space: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    absorber_name: Optional[str] = None
    absorber_eg: float = 1.75
    sq_limit: float = 27.2
    jsc_max: float = 46.0
    cbo: Optional[float] = None
    vbo: Optional[float] = None
    feature_cols: List[str] = field(default_factory=list)

    def get_layer_by_role(self, role: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.role == role), None)


def safe_read_text(path: str) -> str:
    """Read text while preserving compatibility with SCAPS files saved in mixed encodings."""
    raw = Path(path).read_bytes()
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    raise ValueError(f'Cannot read: {path}')


def safe_read_text_with_meta(path: str):
    """Return (text, encoding, newline) so rewritten .scaps files keep original style."""
    raw = Path(path).read_bytes()
    if b'\r\n' in raw:
        newline = '\r\n'
    elif b'\n' in raw:
        newline = '\n'
    else:
        newline = None

    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            text = raw.decode(enc)
            return text, enc, newline
        except Exception:
            continue
    raise ValueError(f'Cannot read: {path}')


def safe_write_text(path: str, content: str, encoding: str = 'utf-8', newline: str = None) -> None:
    """Write text, creating parent folders and preserving original newline style if supplied."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding=encoding, newline='') as f:
        if newline is not None:
            content = content.replace('\r\n', '\n').replace('\n', newline)
        f.write(content)


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _mid_log(lo, hi):
    return 10 ** ((math.log10(lo) + math.log10(hi)) / 2)


def _pct_err(target, got):
    if target is None or got is None:
        return None
    return abs(float(target) - float(got)) / max(abs(float(target)), 1e-30) * 100.0


def _grab_first_float(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        return None
    return float(m.group(1))


# Backward-compatible aliases.
safe_read_text_clean = safe_read_text
safe_read_text_with_meta_clean = safe_read_text_with_meta
safe_write_text_clean = safe_write_text

# Shockley-Queisser limit via interpolation
def sq_limit_interp(eg: float) -> float:
    """SQ limit (%) via interpolation over detailed table."""
    table = {
        0.8: 27.5, 0.9: 31.0, 1.0: 33.9, 1.1: 33.7, 1.2: 33.2,
        1.3: 32.4, 1.34: 33.7, 1.4: 31.4, 1.5: 30.2, 1.6: 29.1,
        1.7: 27.9, 1.75: 27.2, 1.8: 26.8, 1.9: 25.5, 2.0: 24.0,
        2.1: 22.5, 2.2: 21.0, 2.3: 19.5, 2.5: 16.5, 3.0: 8.0,
    }
    xs = sorted(table.keys())
    ys = [table[x] for x in xs]
    f = interpolate.interp1d(xs, ys, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return float(f(eg))

print('Utilities + DeviceConfig ready.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 5 — Thread-safe SQLite trial storage
# ============================================================
# SQLite with a lock is atomic and fast:
#   - auto-creates / auto-adds missing columns when schema evolves
#   - normalizes numpy scalars and NaN values
#   - exposes valid_df() and count(), required by later cells
#   - keeps *_t and *_thickness aliases synchronized for old/new rows
# ============================================================

class TrialStore:
    """Thread-safe trial storage with SQLite backend."""

    BASE_COLUMNS = {
        'seed': 'INTEGER',
        'round_number': 'INTEGER',
        'trial_number': 'INTEGER',
        'is_boundary_eval': 'INTEGER DEFAULT 0',
        'parse_success': 'INTEGER DEFAULT 0',
        'scaps_run_success': 'INTEGER DEFAULT 0',
        'valid_run': 'INTEGER DEFAULT 0',
        'invalid_reason': 'TEXT',
        'Voc': 'REAL',
        'Jsc': 'REAL',
        'FF': 'REAL',
        'eta': 'REAL',
        'runtime_seconds': 'REAL',
        'wall_clock_start': 'REAL',
    }

    def __init__(self, db_path: str, feature_cols: List[str]):
        self.db_path = self._normalize_db_path(db_path)
        self.feature_cols = list(feature_cols or [])
        self._lock = threading.Lock()
        self._init_db()

    @staticmethod
    def _normalize_db_path(db_path: str) -> str:
        if isinstance(db_path, str) and db_path.startswith('sqlite:///'):
            return db_path.replace('sqlite:///', '', 1)
        return str(db_path)

    @staticmethod
    def _normalize_value(v):
        try:
            if isinstance(v, np.generic):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    @staticmethod
    def _column_type(v) -> str:
        v = TrialStore._normalize_value(v)
        if isinstance(v, str):
            return 'TEXT'
        if isinstance(v, bool):
            return 'INTEGER'
        if isinstance(v, int):
            return 'INTEGER'
        return 'REAL'

    def _ensure_parent(self):
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _ensure_table_and_columns(self, conn, row: Optional[dict] = None):
        cols = ['id INTEGER PRIMARY KEY AUTOINCREMENT']
        cols += [f'"{name}" {typ}' for name, typ in self.BASE_COLUMNS.items()]
        cols += [f'"{c}" REAL' for c in self.feature_cols if c not in self.BASE_COLUMNS]
        conn.execute(f'CREATE TABLE IF NOT EXISTS trials ({", ".join(cols)})')

        existing_cols = {r[1] for r in conn.execute('PRAGMA table_info(trials)').fetchall()}
        planned = {**self.BASE_COLUMNS, **{c: 'REAL' for c in self.feature_cols}}
        if row:
            planned.update({k: self._column_type(v) for k, v in row.items()})

        for k, typ in planned.items():
            if k not in existing_cols:
                conn.execute(f'ALTER TABLE trials ADD COLUMN "{k}" {typ}')
                existing_cols.add(k)

    def _init_db(self):
        self._ensure_parent()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table_and_columns(conn)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_seed_round ON trials(seed, round_number)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_valid ON trials(valid_run)')
            conn.commit()

    def insert(self, row: dict):
        clean_row = {k: self._normalize_value(v) for k, v in dict(row).items()}
        with self._lock:
            self._ensure_parent()
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_table_and_columns(conn, clean_row)
                cols = list(clean_row.keys())
                placeholders = ', '.join(['?'] * len(cols))
                col_sql = ', '.join(f'"{c}"' for c in cols)
                values = [clean_row[c] for c in cols]
                conn.execute(f'INSERT INTO trials ({col_sql}) VALUES ({placeholders})', values)
                conn.commit()

    def _postprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        # Synchronize old storage name (*_thickness) with optimizer feature name (*_t).
        layer_names = [getattr(l, 'name', None) for l in globals().get('device', DeviceConfig()).layers]
        for name in [n for n in layer_names if n]:
            t_col = f'{name}_t'
            legacy_col = f'{name}_thickness'
            if legacy_col in df.columns and t_col not in df.columns:
                df[t_col] = df[legacy_col]
            elif legacy_col in df.columns and t_col in df.columns:
                df[t_col] = df[t_col].combine_first(df[legacy_col])
            elif t_col in df.columns and legacy_col not in df.columns:
                df[legacy_col] = df[t_col]

        return df

    def to_dataframe(self, seed: Optional[int] = None) -> pd.DataFrame:
        self._ensure_parent()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table_and_columns(conn)
            if seed is None:
                df = pd.read_sql('SELECT * FROM trials', con=conn)
            else:
                df = pd.read_sql('SELECT * FROM trials WHERE seed = ?', con=conn, params=[int(seed)])
        return self._postprocess_df(df)

    def valid_df(self, seed: Optional[int] = None) -> pd.DataFrame:
        df = self.to_dataframe(seed=seed)
        if df.empty:
            return df
        if 'valid_run' in df.columns:
            df = df[pd.to_numeric(df['valid_run'], errors='coerce').fillna(0).astype(int) == 1]
        if 'eta' in df.columns:
            df = df[pd.to_numeric(df['eta'], errors='coerce').notna()]
        return df.reset_index(drop=True)

    def count(self, seed: Optional[int] = None, valid_only: bool = False) -> int:
        self._ensure_parent()
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table_and_columns(conn)
            clauses = []
            params = []
            if seed is not None:
                clauses.append('seed = ?')
                params.append(int(seed))
            if valid_only:
                clauses.append('valid_run = 1')
            where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
            return int(conn.execute(f'SELECT COUNT(*) FROM trials{where}', params).fetchone()[0])

print('TrialStore ready: auto schema + valid_df/count helpers.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 6 — SCAPS runner (stable /IM kill + mtime monitoring + proven script commands)
# ====================================================================================

class ScapsRunner:
    def __init__(self, scaps_exe: str, working_dir: str,
                 script_file: str, iv_file: str, qe_file: str,
                 timeout: float = 300.0, poll_interval: float = 0.2,
                 settle_seconds: float = 0.6):
        self.scaps_exe = scaps_exe
        self.working_dir = working_dir
        self.script_file = script_file
        self.iv_file = iv_file
        self.qe_file = qe_file
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.settle_seconds = settle_seconds
        self.is_windows = platform.system() == 'Windows'
        self.last_error = ''
        self.last_popup_texts = []

    def _cleanup_outputs(self):
        for f in [self.iv_file, self.qe_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _kill_process(self, proc):
        """
        Kill SCAPS by executable name on Windows.
        This is intentional: popup/dialog child processes often survive /PID killing.
        """
        if proc is None and not self.is_windows:
            return
        try:
            if self.is_windows:
                subprocess.run(
                    'taskkill /F /IM scaps3311.exe',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
            else:
                if proc is None or proc.poll() is not None:
                    return
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass

    def _file_ready(self, path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False

    def _file_mtime(self, path: str):
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return os.path.getmtime(path)
        except Exception:
            pass
        return None

    def _detect_and_dismiss_windows_popups(self):
        hits = []
        if not self.is_windows:
            return hits

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            WM_CLOSE = 0x0010
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

            keywords = [
                'convergence failure',
                'no iv calculation made yet',
                'calculating short circuit',
                'calculation failed',
                'error'
            ]

            def get_window_text(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                return buf.value or ''

            @EnumWindowsProc
            def enum_proc(hwnd, lParam):
                try:
                    if not user32.IsWindowVisible(hwnd):
                        return True
                    title = get_window_text(hwnd).strip()
                    low = title.lower()
                    if any(k in low for k in keywords):
                        hits.append(title)
                        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                except Exception:
                    pass
                return True

            user32.EnumWindows(enum_proc, 0)

        except Exception:
            pass

        return hits

    def run(self, wait_for_qe: bool = False) -> bool:
        """
        Success only if target output file is freshly written and non-empty.
        """
        self.last_error = ''
        self.last_popup_texts = []
        self._cleanup_outputs()

        target_file = self.qe_file if wait_for_qe else self.iv_file
        initial_mtime = self._file_mtime(target_file)

        if self.is_windows:
            cmd = f'"{self.scaps_exe}" "{self.script_file}"'
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.working_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc = subprocess.Popen(
                [self.scaps_exe, self.script_file],
                cwd=self.working_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        saw_popup = False

        try:
            start = time.time()

            while time.time() - start < self.timeout:
                current_mtime = self._file_mtime(target_file)

                # Primary success path: mtime changed + file non-empty
                if current_mtime is not None and current_mtime != initial_mtime and self._file_ready(target_file):
                    time.sleep(self.settle_seconds)
                    if self._file_ready(target_file):
                        return True

                # Secondary safety net: best-effort popup dismissal
                popup_hits = self._detect_and_dismiss_windows_popups()
                if popup_hits:
                    saw_popup = True
                    self.last_popup_texts.extend(popup_hits)

                # Process exited -> decide based on output
                if proc.poll() is not None:
                    if self._file_ready(target_file):
                        time.sleep(min(self.settle_seconds, 0.3))
                        if self._file_ready(target_file):
                            return True

                    if saw_popup:
                        self.last_error = 'popup_convergence_failure'
                    elif os.path.exists(target_file):
                        try:
                            if os.path.getsize(target_file) == 0:
                                self.last_error = 'empty_output_file'
                            else:
                                self.last_error = 'process_exited_without_output'
                        except Exception:
                            self.last_error = 'process_exited_without_output'
                    else:
                        self.last_error = 'process_exited_without_output'
                    return False

                time.sleep(self.poll_interval)

            # Final timeout fallback
            if self._file_ready(target_file):
                time.sleep(min(self.settle_seconds, 0.3))
                if self._file_ready(target_file):
                    return True

            self.last_error = 'popup_convergence_failure_timeout' if saw_popup else 'timeout_waiting_for_output'
            return False

        finally:
            self._kill_process(proc)


# --- IV/QE sampling parameters ---
IV_START_V      = 0.0
IV_STOP_V       = 2.0
IV_INCREMENT_V  = 0.005
IV_POINTS       = int(round((IV_STOP_V - IV_START_V) / IV_INCREMENT_V)) + 1

QE_START_NM     = 200
QE_STOP_NM      = 900
QE_INCREMENT_NM = 5
QE_POINTS       = int(round((QE_STOP_NM - QE_START_NM) / QE_INCREMENT_NM)) + 1


def write_scaps_script(script_file: str, load_file: str,
                       iv_file: str, qe_file: str,
                       run_qe: bool = False,
                       temperature_k: Optional[float] = None,
                       rs_val_ohm_cm2: Optional[float] = None,
                       rsh_val_ohm_cm2: Optional[float] = None,
                       use_sq_limit: bool = False,
                       eb_file: Optional[str] = None,
                       gen_file: Optional[str] = None,
                       rec_file: Optional[str] = None) -> None:
    """
    Stable SCAPS script writer using the proven command family from the smooth notebook.
    """
    lines = [
        f'load allscapssettingsfile {load_file}',
        'set script_display_mode.fully_suppressed',
    ]

    if use_sq_limit:
        lines.append('set numerical.Shockley-Queisser_limit if_needed')

    if temperature_k is not None:
        lines.append(f'action workingpoint.temperature {temperature_k}')

    if rs_val_ohm_cm2 is not None:
        lines.append(f'set external.Rs {rs_val_ohm_cm2:.6e}')
    if rsh_val_ohm_cm2 is not None:
        lines.append(f'set external.Rsh {rsh_val_ohm_cm2:.6e}')

    lines += [
        'action light',
        f'action iv.startv {IV_START_V}',
        f'action iv.stopv {IV_STOP_V}',
        f'action iv.increment {IV_INCREMENT_V}',
        f'action iv.points {IV_POINTS}',
        'action iv.checkaction 1',
    ]

    if run_qe:
        lines += [
            'action iv.stopaftervoc 0',   # continue after Voc for QE
            'calculate singleshot',
            f'save results.iv {iv_file}',
            f'action qe.startlambda {QE_START_NM}',
            f'action qe.stoplambda {QE_STOP_NM}',
            f'action qe.increment {QE_INCREMENT_NM}',
            f'action qe.points {QE_POINTS}',
            'action qe.doqe',
            f'save results.qe {qe_file}',
        ]
    else:
        lines += [
            'action iv.stopaftervoc 1',   # stop after Voc for normal IV
            'calculate singleshot',
            f'save results.iv {iv_file}',
        ]

    if eb_file is not None:
        lines.append(f'save results.eb {eb_file}')
    if gen_file is not None:
        lines.append(f'save results.gen {gen_file}')
    if rec_file is not None:
        lines.append(f'save results.rec {rec_file}')

    lines.append('set quitscript.quitscaps')
    safe_write_text(script_file, '\n'.join(lines) + '\n')



_FLOAT_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

def _first_metric(text: str, patterns: List[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _numeric_rows_from_text(text: str, min_cols: int = 2) -> List[List[float]]:
    rows = []
    for line in text.splitlines():
        clean = line.strip().replace(',', ' ')
        if not clean or clean.startswith(('#', ';', '//')):
            continue
        vals = re.findall(_FLOAT_RE, clean)
        if len(vals) >= min_cols:
            try:
                rows.append([float(v) for v in vals])
            except Exception:
                continue
    return rows


def _interp_zero_crossing(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return None
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    exact = np.where(np.isclose(y, 0.0, atol=max(np.nanmax(np.abs(y)) * 1e-6, 1e-12)))[0]
    if len(exact):
        return float(x[exact[-1]])

    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == y2:
            continue
        if (y1 <= 0 <= y2) or (y2 <= 0 <= y1):
            return float(x[i] - y1 * (x[i + 1] - x[i]) / (y2 - y1))

    # Some SCAPS tables approach zero without changing sign.
    idx = int(np.nanargmin(np.abs(y)))
    if abs(y[idx]) <= 0.05 * max(np.nanmax(np.abs(y)), 1e-30):
        return float(x[idx])
    return None


def parse_iv(iv_file: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse SCAPS IV output and return (Voc, Jsc, FF, eta).

    The parser first looks for explicit named metrics in the file. If those are
    unavailable, it falls back to a numeric J-V table and estimates Voc, Jsc,
    FF, and PCE from the curve. Jsc is returned as a positive mA/cm² value and
    FF/PCE are returned in percent units.
    """
    if not os.path.exists(iv_file) or os.path.getsize(iv_file) == 0:
        return None

    try:
        text = safe_read_text(iv_file)
    except Exception:
        with open(iv_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

    voc = _first_metric(text, [
        rf'\bV\s*oc\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'\bVoc\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'open\s*circuit\s*voltage[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
    ])
    jsc = _first_metric(text, [
        rf'\bJ\s*sc\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'\bJsc\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'short\s*circuit\s*current[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
    ])
    ff = _first_metric(text, [
        rf'\bFF\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'fill\s*factor[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
    ])
    eta = _first_metric(text, [
        rf'\bPCE\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'\beta\b[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
        rf'efficiency[^:=\n\r]*[:=]\s*({_FLOAT_RE})',
    ])

    if ff is not None and 0 < ff <= 1.5:
        ff *= 100.0
    if eta is not None and 0 < eta <= 1.5:
        eta *= 100.0
    if jsc is not None:
        jsc = abs(jsc)

    if voc is not None and jsc is not None and ff is not None and eta is None:
        eta = voc * jsc * ff / 100.0
    if voc is not None and jsc is not None and eta is not None and ff is None and voc * jsc > 0:
        ff = eta / (voc * jsc) * 100.0

    named = [voc, jsc, ff, eta]
    if all(v is not None and np.isfinite(v) for v in named):
        return float(voc), float(abs(jsc)), float(ff), float(eta)

    # Fallback: infer from a numeric J-V table.
    rows = _numeric_rows_from_text(text, min_cols=2)
    if len(rows) < 3:
        return None
    width = max(len(r) for r in rows)
    arr = np.array([r + [np.nan] * (width - len(r)) for r in rows], dtype=float)
    v = arr[:, 0]

    # Choose a current column. Prefer one with a zero crossing or near-zero endpoint.
    best_j = None
    best_score = -1.0
    for col_idx in range(1, arr.shape[1]):
        j_candidate = arr[:, col_idx]
        finite = np.isfinite(v) & np.isfinite(j_candidate)
        if finite.sum() < 3:
            continue
        jj = j_candidate[finite]
        score = float(np.nanstd(jj))
        if np.nanmin(jj) <= 0 <= np.nanmax(jj):
            score += 1e6
        if score > best_score:
            best_score = score
            best_j = j_candidate
    if best_j is None:
        return None

    finite = np.isfinite(v) & np.isfinite(best_j)
    v = v[finite]
    j = best_j[finite]
    if len(v) < 3:
        return None
    order = np.argsort(v)
    v = v[order]
    j = j[order]

    try:
        j_at_zero = float(np.interp(0.0, v, j)) if v.min() <= 0 <= v.max() else float(j[np.argmin(np.abs(v))])
        jsc_curve = abs(j_at_zero)
        voc_curve = _interp_zero_crossing(v, j)
        p_pos = np.nanmax(v * j)
        p_neg = np.nanmax(-v * j)
        eta_curve = float(max(p_pos, p_neg))
        if voc_curve is None or voc_curve <= 0 or jsc_curve <= 0 or eta_curve <= 0:
            return None
        ff_curve = eta_curve / (voc_curve * jsc_curve) * 100.0
        if not all(np.isfinite(x) for x in [voc_curve, jsc_curve, ff_curve, eta_curve]):
            return None
        return float(voc_curve), float(jsc_curve), float(ff_curve), float(eta_curve)
    except Exception:
        return None

print('SCAPS runner ready: /IM kill + mtime monitoring + stable script commands + parse_iv')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 7 — Auto-detect layers and interfaces from .scaps file
# ============================================================
VALID_ROLES = {'TCO', 'ETL', 'HTL', 'ABSORBER', 'BUFFER', 'BSF', 'METAL'}

def extract_layer_blocks(scaps_file: str) -> List[dict]:
    text = safe_read_text(scaps_file)
    parts = re.split(r'(?:^|\r?\n)layer\r?\n', text)
    blocks = []
    for part in parts[1:]:
        nm = re.search(r'name\s*:\s*(.+)', part)
        if nm:
            blocks.append({
                'block_name': nm.group(1).strip().replace('\r', ''),
                'raw_block': part
            })
    return blocks

def infer_role(block_name: str) -> str:
    if block_name in MANUAL_ROLES:
        return MANUAL_ROLES[block_name]
    prefix = block_name.split('_', 1)[0].strip().upper()
    return prefix if prefix in VALID_ROLES else 'OTHER'

def infer_doping_type(raw_block: str) -> str:
    na = re.search(r'Na\(uniform\)\s*:\t\s*([0-9Ee\+\-\.]+)', raw_block)
    nd = re.search(r'Nd\(uniform\)\s*:\t\s*([0-9Ee\+\-\.]+)', raw_block)
    na_val = float(na.group(1)) if na else 0.0
    nd_val = float(nd.group(1)) if nd else 0.0
    if   na_val > 1e5 and nd_val <= 1e5: return 'p'
    elif nd_val > 1e5 and na_val <= 1e5: return 'n'
    elif na_val > 1e5 and nd_val > 1e5:  return 'both'
    return 'n'

def infer_thickness_nm(raw_block: str) -> Optional[float]:
    m = re.search(r'^d\s*:\s*([0-9Ee\+\-\.]+)', raw_block, re.MULTILINE)
    return float(m.group(1)) * 1e9 if m else None

def detect_interfaces(scaps_file: str) -> List[str]:
    text = safe_read_text(scaps_file)
    ifaces = re.findall(r'interfacename\s*:\s*(.+)', text)
    return [i.strip().replace('\r', '').replace('\n', '')
            for i in ifaces if '/' in i]

def _active_param_ranges_for_role(role: str, ranges_source: Optional[dict] = None) -> Dict[str, Tuple[float, float]]:
    """Return only real optimizable ranges for a role (skip empty/None placeholders)."""
    source = ranges_source if ranges_source is not None else INITIAL_PARAM_RANGES
    raw = source.get(role, {})
    active = {}
    for key, value in raw.items():
        if isinstance(value, tuple) and len(value) == 2 and all(v is not None for v in value):
            active[key] = value
    return active

def _normalize_iface_name(name: str) -> str:
    return re.sub(r'\s+', '', name).strip()

TARGET_INTERFACE_NAMES = {'CBTS/MASnIBr2', 'MASnIBr2/WO3'}

def build_device_config(scaps_file: str) -> DeviceConfig:
    cfg = DeviceConfig()

    blocks = extract_layer_blocks(scaps_file)
    for b in blocks:
        bn = b['block_name']
        role = infer_role(bn)
        active_ranges = _active_param_ranges_for_role(role)
        cfg.layers.append(Layer(
            name=bn,
            block_name=bn,
            role=role,
            doping_type=infer_doping_type(b['raw_block']),
            thickness_nm=infer_thickness_nm(b['raw_block']),
            optimize=(role not in ('METAL', 'OTHER') and len(active_ranges) > 0),
        ))

    cfg.interfaces = detect_interfaces(scaps_file)
    if OPTIMIZE_INTERFACE_DEFECTS:
        for iface in cfg.interfaces:
            norm = _normalize_iface_name(iface)
            if norm in TARGET_INTERFACE_NAMES:
                key = f'iface_{re.sub(r"[^a-zA-Z0-9]", "_", norm)}_Nt'
                cfg.interface_space[key] = INTERFACE_RANGE

    # Absorber bandgap
    abs_layer = cfg.get_layer_by_role('ABSORBER')
    if abs_layer:
        cfg.absorber_name = abs_layer.name
        cfg.absorber_eg = ABSORBER_BANDGAP_MAP.get(abs_layer.name, 1.75)

    # SQ limit
    cfg.sq_limit = sq_limit_interp(cfg.absorber_eg)

    # Band alignment (display only)
    # Band-offset convention:
    #   CBO = chi_absorber - chi_ETL
    #   VBO = chi_HTL - chi_absorber + Eg_HTL - Eg_absorber
    etl = cfg.get_layer_by_role('ETL')
    abs_l = cfg.get_layer_by_role('ABSORBER')
    htl = cfg.get_layer_by_role('HTL')
    if etl and abs_l and etl.name in MATERIAL_PARAMS and abs_l.name in MATERIAL_PARAMS:
        cfg.cbo = MATERIAL_PARAMS[abs_l.name]['chi'] - MATERIAL_PARAMS[etl.name]['chi']
    if abs_l and htl and abs_l.name in MATERIAL_PARAMS and htl.name in MATERIAL_PARAMS:
        cfg.vbo = ((MATERIAL_PARAMS[htl.name]['chi'] - MATERIAL_PARAMS[abs_l.name]['chi']) +
                   (MATERIAL_PARAMS[htl.name]['Eg'] - MATERIAL_PARAMS[abs_l.name]['Eg']))

    # Feature columns
    for l in cfg.layers:
        if not l.optimize: 
            continue
        r = _active_param_ranges_for_role(l.role)
        if 'thickness' in r:                                cfg.feature_cols.append(f'{l.name}_t')
        if 'defect' in r:                                   cfg.feature_cols.append(f'{l.name}_defect')
        if 'NA' in r and l.doping_type in ('p', 'both'):    cfg.feature_cols.append(f'{l.name}_NA')
        if 'ND' in r and l.doping_type in ('n', 'both'):    cfg.feature_cols.append(f'{l.name}_ND')

    for key in cfg.interface_space:
        cfg.feature_cols.append(key)
    cfg.feature_cols.append('back_contact_wf')

    return cfg


def rebuild_feature_cols_from_current_device(dev: DeviceConfig) -> List[str]:
    """Rebuild feature columns after template/interface fixes."""
    cols = []
    for l in dev.layers:
        if not l.optimize:
            continue
        r = _active_param_ranges_for_role(l.role, INITIAL_PARAM_RANGES)
        bn = l.block_name
        if 'thickness' in r and r['thickness'] is not None:
            cols.append(f'{bn}_t')
        if 'defect' in r and r['defect'] is not None:
            cols.append(f'{bn}_defect')
        if 'NA' in r and r['NA'] is not None and l.doping_type in ('p', 'both'):
            cols.append(f'{bn}_NA')
        if 'ND' in r and r['ND'] is not None and l.doping_type in ('n', 'both'):
            cols.append(f'{bn}_ND')
    for k in dev.interface_space.keys():
        cols.append(k)
    cols.append('back_contact_wf')
    return cols


# Build the device from the user-prepared template (do NOT overwrite it here)


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 8 — .scaps file builder with round-trip verification
# ============================================================
# Validated .scaps file builder:
#   - updates layer parameters in raw SCAPS units
#   - updates interface N in raw SCAPS units
#   - aligns interface names to the real template order
#   - fixes front contact at VALIDATED_FRONT_WF
#   - writes back contact from back_contact_wf (default VALIDATED_BACK_WF)
#   - verifies every write by reading the generated .scaps file back
# ============================================================

_LAYER_STOP_RE = (
    r'^\s*interfacename\s*:|'
    r'^\s*name\s*:|'
    r'^\s*front\s+contact|'
    r'^\s*back\s+contact|'
    r'\Z'
)


def _extract_layer_block(text: str, block_name: str):
    escaped = re.escape(block_name)
    pattern = rf'(^\s*name\s*:\s*{escaped}\b.*?)(?={_LAYER_STOP_RE})'
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1) if m else None


def _extract_interface_block(text: str, iface_name: str):
    tokens = [t.strip() for t in iface_name.split('/')]
    iface_pattern = r'\s*/\s*'.join(re.escape(tok) for tok in tokens)
    pattern = rf'(^\s*interfacename\s*:\s*{iface_pattern}\s*.*?)(?={_LAYER_STOP_RE})'
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1) if m else None


# Backward-compatible aliases.
_extract_layer_block_clean = _extract_layer_block
_extract_interface_block_clean = _extract_interface_block


def _iface_key_to_template_name(key: str) -> str:
    """Convert iface_CBTS_MASnIBr2_Nt -> CBTS/MASnIBr2."""
    raw = key
    if raw.startswith('iface_'):
        raw = raw[len('iface_'):]
    if raw.endswith('_Nt'):
        raw = raw[:-len('_Nt')]
    return raw.replace('_', '/')


def set_contact_value_by_block(text: str, which_block: str, value: float) -> str:
    pattern = rf'(?is)({which_block}\s*contact.*?Fi_m\s*:\s*)[\d\.eE\+\-]+'
    new_text, n = re.subn(
        pattern,
        lambda m: m.group(1) + f'{float(value):.6f}',
        text,
        count=1,
    )
    if n == 0:
        raise ValueError(f'Could not set {which_block} contact Fi_m')
    return new_text


def _set_contact_work_function(text: str, back_contact_wf: float) -> str:
    """Backward-compatible helper: update only the back contact."""
    return set_contact_value_by_block(text, 'back', back_contact_wf)


# Backward-compatible alias.
_set_contact_work_function_clean = _set_contact_work_function


def _replace_first_in_block(block: str, pattern: str, replacement, label: str, block_name: str) -> str:
    new_block, n = re.subn(pattern, replacement, block, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f'{label} no-match in {block_name}')
    return new_block


def build_updated_scaps_file(
    template_path: str,
    dest_path: str,
    layer_updates: dict,
    iface_vals: dict,
    back_contact_wf: float = None,
    verify: bool = True,
) -> None:
    text, src_enc, src_newline = safe_read_text_with_meta(template_path)

    # -------------------------
    # Layer updates
    # -------------------------
    for bn, updates in layer_updates.items():
        escaped = re.escape(bn)
        pattern = rf'(^\s*name\s*:\s*{escaped}\b.*?)(?={_LAYER_STOP_RE})'
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if not m:
            raise ValueError(f'Layer block not found: {bn}')

        old_block = m.group(1)
        new_block = old_block

        if updates.get('thickness_um') is not None:
            t_m = float(updates['thickness_um']) * 1e-6
            new_block = _replace_first_in_block(
                new_block,
                r'(^\s*d\s*:\s*)\S+',
                lambda mm: mm.group(1) + f'{t_m:.6e}',
                'Thickness', bn,
            )

        if updates.get('na_val') is not None:
            target_na_m3 = float(updates['na_val']) * 1e6
            new_block = _replace_first_in_block(
                new_block,
                r'(Na\(uniform\)\s*:\s*)\S+',
                lambda mm: mm.group(1) + f'{target_na_m3:.6e}',
                'Na', bn,
            )

        if updates.get('nd_val') is not None:
            target_nd_m3 = float(updates['nd_val']) * 1e6
            new_block = _replace_first_in_block(
                new_block,
                r'(Nd\(uniform\)\s*:\s*)\S+',
                lambda mm: mm.group(1) + f'{target_nd_m3:.6e}',
                'Nd', bn,
            )

        if updates.get('defect') is not None:
            target_nt_m3 = float(updates['defect']) * 1e6
            new_block = _replace_first_in_block(
                new_block,
                r'(Nt\(uniform\)\s*:\s*)\S+',
                lambda mm: mm.group(1) + f'{target_nt_m3:.6e}',
                'Defect Nt', bn,
            )

        text = _replace_span(text, m.start(1), m.end(1), new_block)

    # -------------------------
    # Interface updates: cm^-2 -> m^-2
    # -------------------------
    for key, val in iface_vals.items():
        iface_name = _iface_key_to_template_name(key)
        tokens = [t.strip() for t in iface_name.split('/')]
        iface_pattern = r'\s*/\s*'.join(re.escape(tok) for tok in tokens)
        pattern = rf'(^\s*interfacename\s*:\s*{iface_pattern}\s*.*?)(?={_LAYER_STOP_RE})'
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if not m:
            raise ValueError(f'Interface block not found: {iface_name}')

        old_block = m.group(1)
        new_block = _replace_first_in_block(
            old_block,
            r'(^\s*N\s*:\s*)\S+',
            lambda mm: mm.group(1) + f'{float(val) * 1e4:.6e}',
            'Interface N', iface_name,
        )
        text = _replace_span(text, m.start(1), m.end(1), new_block)

    # -------------------------
    # Validated contacts
    # -------------------------
    front_wf = globals().get('VALIDATED_FRONT_WF', 4.002469)
    if back_contact_wf is None:
        back_contact_wf = globals().get('VALIDATED_BACK_WF', 5.60)
    text = set_contact_value_by_block(text, 'front', front_wf)
    text = set_contact_value_by_block(text, 'back', back_contact_wf)

    safe_write_text(dest_path, text, src_enc, src_newline)

    # -------------------------
    # Round-trip verification
    # -------------------------
    if verify:
        written_text = safe_read_text(dest_path)

        for bn, updates in layer_updates.items():
            blk = _extract_layer_block(written_text, bn)
            if blk is None:
                raise AssertionError(f'Layer missing after write: {bn}')

            if updates.get('thickness_um') is not None:
                raw_d = _grab_first_float(r'^\s*d\s*:\s*([\d\.eE\+\-]+)', blk, flags=re.MULTILINE)
                if raw_d is None:
                    raise AssertionError(f'{bn} thickness not found after write')
                got_um = raw_d * 1e6
                if _pct_err(updates['thickness_um'], got_um) > 1.0:
                    raise AssertionError(f'{bn} thickness mismatch: target {updates["thickness_um"]:.6e}, got {got_um:.6e}')

            if updates.get('na_val') is not None:
                raw_na = _grab_first_float(r'Na\(uniform\)\s*:\s*([\d\.eE\+\-]+)', blk)
                if raw_na is None:
                    raise AssertionError(f'{bn} Na not found after write')
                got_cm3 = raw_na / 1e6
                if _pct_err(updates['na_val'], got_cm3) > 1.0:
                    raise AssertionError(f'{bn} Na mismatch: target {updates["na_val"]:.6e}, got {got_cm3:.6e}')

            if updates.get('nd_val') is not None:
                raw_nd = _grab_first_float(r'Nd\(uniform\)\s*:\s*([\d\.eE\+\-]+)', blk)
                if raw_nd is None:
                    raise AssertionError(f'{bn} Nd not found after write')
                got_cm3 = raw_nd / 1e6
                if _pct_err(updates['nd_val'], got_cm3) > 1.0:
                    raise AssertionError(f'{bn} Nd mismatch: target {updates["nd_val"]:.6e}, got {got_cm3:.6e}')

            if updates.get('defect') is not None:
                raw_nt = _grab_first_float(r'Nt\(uniform\)\s*:\s*([\d\.eE\+\-]+)', blk)
                if raw_nt is None:
                    raise AssertionError(f'{bn} Nt not found after write')
                got_cm3 = raw_nt / 1e6
                if _pct_err(updates['defect'], got_cm3) > 1.0:
                    raise AssertionError(f'{bn} defect mismatch: target {updates["defect"]:.6e}, got {got_cm3:.6e}')

        for key, target_cm2 in iface_vals.items():
            iface_name = _iface_key_to_template_name(key)
            blk = _extract_interface_block(written_text, iface_name)
            if blk is None:
                raise AssertionError(f'Interface missing after write: {iface_name}')
            raw_n = _grab_first_float(r'^\s*N\s*:\s*([\d\.eE\+\-]+)', blk, flags=re.MULTILINE)
            if raw_n is None:
                raise AssertionError(f'{iface_name} N not found after write')
            got_cm2 = raw_n / 1e4
            if _pct_err(target_cm2, got_cm2) > 1.0:
                raise AssertionError(f'{iface_name} interface N mismatch: target {target_cm2:.6e}, got {got_cm2:.6e}')

        front = _grab_first_float(r'(?is)front\s+contact.*?Fi_m\s*:\s*([\d\.eE\+\-]+)', written_text)
        back = _grab_first_float(r'(?is)back\s+contact.*?Fi_m\s*:\s*([\d\.eE\+\-]+)', written_text)
        if front is None or abs(front - front_wf) / max(abs(front_wf), 1e-30) > 1e-6:
            raise AssertionError(f'Front contact mismatch: target {front_wf}, got {front}')
        if back is None or abs(back - float(back_contact_wf)) / max(abs(float(back_contact_wf)), 1e-30) > 1e-6:
            raise AssertionError(f'Back contact mismatch: target {back_contact_wf}, got {back}')


# Backward-compatible alias.
build_updated_scaps_file_clean = build_updated_scaps_file

print('Validated .scaps builder ready: layers + interfaces + contacts + round-trip verification.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 9 — Physics precheck (4 constraints)
# =====================================================================
# Four constraints:
#   C1: Diffusion length (absorber thickness ≤ 3×L_diff based on Nt)
#   C2: Non-degenerate doping (doping ≤ DOS boundary, Boltzmann validity)
#   C3: Junction formation (ETL/HTL doping > absorber effective)
#   C4: Thin-film limit (layers ≥ 20 nm, absorber ≥ 50 nm)
#
# Important implementation notes:
#   - Uses the current WO3 / MASnIBr2 / CBTS stack
#   - Accepts both internal optimizer keys (nd_val/na_val) and legacy keys (ND/NA)
#   - Self-tests disable legacy CBO/VBO range checks so each constraint is tested in isolation
# =====================================================================

import copy

# --- Material properties for precheck calculations ---
PRECHECK_MATERIAL_PROPS = {
    'masnibr2': {
        'Eg'    : 1.75,
        'chi'   : 3.78,
        'Nc'    : 1.0e18,
        'Nv'    : 1.0e18,
        'mu_n'  : 1.6,
        'mu_p'  : 1.6,
        'sigma' : 1.0e-15,
        'v_th'  : 1.0e7,
    },
    'wo3': {
        'Eg'    : 2.00,
        'chi'   : 3.80,
        'Nc'    : 2.2e21,
        'Nv'    : 2.2e21,
        'mu_n'  : 30.0,
        'mu_p'  : 30.0,
        'sigma' : 1.0e-15,
        'v_th'  : 1.0e7,
    },
    'cbts': {
        'Eg'    : 1.90,
        'chi'   : 3.60,
        'Nc'    : 2.2e18,
        'Nv'    : 1.8e19,
        'mu_n'  : 30.0,
        'mu_p'  : 10.0,
        'sigma' : 1.0e-15,
        'v_th'  : 1.0e7,
    },
}

PRECHECK_CONSTANTS = {
    'kT_q_300K'               : 0.02585,
    'absorber_eff_carrier'    : 1.0e15,
    'degenerate_threshold'    : 1.0,      # relaxed to DOS boundary
    'diff_length_safety'      : 3.0,
    'thin_film_limit_um'      : 0.020,
    'absorber_min_um'         : 0.050,
}

def _extract_update(updates, *keys):
    """Return first present key from updates, otherwise None."""
    for k in keys:
        if k in updates and updates[k] is not None:
            return updates[k]
    return None

def _compute_diffusion_length_um(Nt_cm3, material='masnibr2'):
    """Compute SRH diffusion length from defect density."""
    if Nt_cm3 is None or Nt_cm3 <= 0:
        return float('inf')

    props = PRECHECK_MATERIAL_PROPS[material]
    D_cm2_s = props['mu_n'] * PRECHECK_CONSTANTS['kT_q_300K']
    tau_s = 1.0 / (Nt_cm3 * props['sigma'] * props['v_th'])
    L_diff_cm = (D_cm2_s * tau_s) ** 0.5
    return L_diff_cm * 1e4

def _check_C1_diffusion_length(absorber_d_um, absorber_Nt):
    if absorber_d_um is None or absorber_Nt is None:
        return {'valid': True, 'skipped': True, 'reason': 'Params missing'}

    L_diff_um = _compute_diffusion_length_um(absorber_Nt, 'masnibr2')
    max_allowed = PRECHECK_CONSTANTS['diff_length_safety'] * L_diff_um

    if absorber_d_um > max_allowed:
        return {
            'valid': False,
            'constraint': 'C1_diffusion_length',
            'reason': (f'Absorber d={absorber_d_um:.3f}μm > 3×L_diff={max_allowed:.3f}μm '
                       f'(Nt={absorber_Nt:.1e})'),
            'L_diff_um': L_diff_um,
            'violation_ratio': absorber_d_um / max(max_allowed, 1e-10),
        }
    return {'valid': True, 'L_diff_um': L_diff_um}

def _check_C2_degenerate_doping(wo3_ND, cbts_NA):
    threshold = PRECHECK_CONSTANTS['degenerate_threshold']

    if wo3_ND is not None:
        wo3_Nc = PRECHECK_MATERIAL_PROPS['wo3']['Nc']
        if wo3_ND > threshold * wo3_Nc:
            return {
                'valid': False,
                'constraint': 'C2_degenerate_doping',
                'reason': f'WO3 Nd={wo3_ND:.2e} > {threshold:.1f}×Nc={threshold*wo3_Nc:.2e} (degenerate)',
            }

    if cbts_NA is not None:
        cbts_Nv = PRECHECK_MATERIAL_PROPS['cbts']['Nv']
        if cbts_NA > threshold * cbts_Nv:
            return {
                'valid': False,
                'constraint': 'C2_degenerate_doping',
                'reason': f'CBTS Na={cbts_NA:.2e} > {threshold:.1f}×Nv={threshold*cbts_Nv:.2e} (degenerate)',
            }

    return {'valid': True}

def _check_C3_junction_formation(wo3_ND, cbts_NA):
    abs_eff = PRECHECK_CONSTANTS['absorber_eff_carrier']

    if wo3_ND is not None and wo3_ND <= abs_eff:
        return {
            'valid': False,
            'constraint': 'C3_junction_formation',
            'reason': f'WO3 Nd={wo3_ND:.2e} ≤ abs_eff={abs_eff:.0e} (no junction)',
        }

    if cbts_NA is not None and cbts_NA <= abs_eff:
        return {
            'valid': False,
            'constraint': 'C3_junction_formation',
            'reason': f'CBTS Na={cbts_NA:.2e} ≤ abs_eff={abs_eff:.0e} (no junction)',
        }

    return {'valid': True}

def _check_C4_thin_film_limit(wo3_d, abs_d, cbts_d):
    limit = PRECHECK_CONSTANTS['thin_film_limit_um']
    abs_min = PRECHECK_CONSTANTS['absorber_min_um']

    checks = [
        ('WO3', wo3_d, limit),
        ('CBTS', cbts_d, limit),
        ('MASnIBr₂', abs_d, abs_min),
    ]

    for name, thickness, min_val in checks:
        if thickness is not None and thickness < min_val:
            return {
                'valid': False,
                'constraint': 'C4_thin_film_limit',
                'reason': f'{name} d={thickness:.3f}μm < {min_val:.3f}μm (sub-continuum)',
            }

    return {'valid': True}

def physics_precheck(layer_updates, iface_vals, device):
    result = physics_precheck_detailed(layer_updates, iface_vals, device)
    return result['valid'], result['reason']

def physics_precheck_detailed(layer_updates, iface_vals, device, apply_legacy_band_checks=True):
    """
    Detailed 4-constraint physics precheck with structured results.
    """
    if not USE_PHYSICS_PRECHECK:
        return {
            'valid': True,
            'reason': 'Pre-check disabled (ablation)',
            'failed_constraints': [],
            'details': {},
        }

    abs_d = abs_Nt = None
    wo3_d = wo3_ND = None
    cbts_d = cbts_NA = None

    for layer in device.layers:
        block_name = layer.block_name
        role = layer.role
        if block_name not in layer_updates:
            continue

        updates = layer_updates[block_name]

        if role == 'ABSORBER':
            abs_d = _extract_update(updates, 'thickness_um', 'thickness')
            abs_Nt = _extract_update(updates, 'defect', 'Nt')
        elif role == 'ETL':
            wo3_d = _extract_update(updates, 'thickness_um', 'thickness')
            wo3_ND = _extract_update(updates, 'nd_val', 'ND', 'Nd')
        elif role == 'HTL':
            cbts_d = _extract_update(updates, 'thickness_um', 'thickness')
            cbts_NA = _extract_update(updates, 'na_val', 'NA', 'Na')

    details = {}
    failed = []

    if PRECHECK_DIFFUSION_LENGTH:
        details['C1_diffusion_length'] = _check_C1_diffusion_length(abs_d, abs_Nt)
        if not details['C1_diffusion_length']['valid']:
            failed.append('C1_diffusion_length')

    if PRECHECK_DEGENERATE_DOPING:
        details['C2_degenerate_doping'] = _check_C2_degenerate_doping(wo3_ND, cbts_NA)
        if not details['C2_degenerate_doping']['valid']:
            failed.append('C2_degenerate_doping')

    if PRECHECK_JUNCTION_FORMATION:
        details['C3_junction_formation'] = _check_C3_junction_formation(wo3_ND, cbts_NA)
        if not details['C3_junction_formation']['valid']:
            failed.append('C3_junction_formation')

    if PRECHECK_THIN_FILM_LIMIT:
        details['C4_thin_film_limit'] = _check_C4_thin_film_limit(wo3_d, abs_d, cbts_d)
        if not details['C4_thin_film_limit']['valid']:
            failed.append('C4_thin_film_limit')

    # Legacy band-offset checks kept for production, but optional in self-tests
    if apply_legacy_band_checks:
        if device.cbo is not None and not (CBO_RANGE[0] <= device.cbo <= CBO_RANGE[1]):
            failed.append('CBO_out_of_range')
            details['CBO_out_of_range'] = {
                'valid': False,
                'reason': f'CBO={device.cbo:+.3f} out of {CBO_RANGE}'
            }

        if device.vbo is not None and not (VBO_RANGE[0] <= device.vbo <= VBO_RANGE[1]):
            failed.append('VBO_out_of_range')
            details['VBO_out_of_range'] = {
                'valid': False,
                'reason': f'VBO={device.vbo:+.3f} out of {VBO_RANGE}'
            }

    if ENFORCE_ABSORBER_THICKEST:
        if abs_d and wo3_d and abs_d < wo3_d:
            failed.append('absorber_thickness_vs_ETL')
            details['absorber_thickness_vs_ETL'] = {
                'valid': False,
                'reason': f'Absorber ({abs_d:.3f}μm) thinner than ETL ({wo3_d:.3f}μm)',
            }
        if abs_d and cbts_d and abs_d < cbts_d:
            failed.append('absorber_thickness_vs_HTL')
            details['absorber_thickness_vs_HTL'] = {
                'valid': False,
                'reason': f'Absorber ({abs_d:.3f}μm) thinner than HTL ({cbts_d:.3f}μm)',
            }

    is_valid = len(failed) == 0
    if is_valid:
        reason = 'All physics constraints passed'
    else:
        reasons = []
        for c in failed:
            if c in details and 'reason' in details[c]:
                reasons.append(details[c]['reason'])
        reason = ' | '.join(reasons) if reasons else 'Unknown violation'

    return {
        'valid': is_valid,
        'reason': reason,
        'failed_constraints': failed,
        'details': details,
    }

def physics_postcheck(Voc, Jsc, FF, eta, device):
    """Post-SCAPS validation."""
    if not USE_PHYSICS_POSTCHECK:
        if any(v is None for v in [Voc, Jsc, FF, eta]):
            return False, 'Missing metric'
        if any(np.isnan(v) or np.isinf(v) for v in [Voc, Jsc, FF, eta]):
            return False, 'NaN/Inf metric'
        return True, 'Post-check disabled (ablation)'

    if any(v is None for v in [Voc, Jsc, FF, eta]):
        return False, 'Missing metric'
    if any(np.isnan(v) or np.isinf(v) for v in [Voc, Jsc, FF, eta]):
        return False, 'NaN/Inf metric'
    if any(v <= 0 for v in [Voc, Jsc, FF, eta]):
        return False, 'Non-positive metric'
    if Jsc > device.jsc_max:
        return False, f'Jsc={Jsc:.2f} > {device.jsc_max}'
    if eta > device.sq_limit:
        return False, f'PCE={eta:.2f} > SQ={device.sq_limit:.1f}%'
    if Voc >= device.absorber_eg:
        return False, f'Voc={Voc:.3f} >= Eg={device.absorber_eg}'

    voc_norm = Voc / 0.02585
    if voc_norm > 1:
        ff_ideal = (voc_norm - np.log(voc_norm + 0.72)) / (voc_norm + 1) * 100
        if FF > ff_ideal + 5:
            return False, f'FF={FF:.1f}% > ideal+5 {ff_ideal+5:.1f}%'

    eta_calc = Voc * Jsc * FF / 100
    if abs(eta_calc - eta) > 2.0:
        return False, f'PCE mismatch: {eta_calc:.3f} vs {eta:.3f}'
    return True, 'Valid'

def _test_physics_precheck():
    """Self-test with 6 cases. Returns True if all pass."""
    from types import SimpleNamespace

    mock_abs = SimpleNamespace(role='ABSORBER', block_name='MASnIBr2')
    mock_etl = SimpleNamespace(role='ETL', block_name='WO3')
    mock_htl = SimpleNamespace(role='HTL', block_name='CBTS')
    mock_dev = SimpleNamespace(
        layers=[mock_etl, mock_abs, mock_htl],
        cbo=-0.02, vbo=-0.03,
    )

    passed = 0
    failed = 0

    valid = {
        'WO3'     : {'thickness_um': 0.05, 'nd_val': 1e18, 'defect': 1e14},
        'MASnIBr2': {'thickness_um': 0.35, 'defect': 1e14},
        'CBTS'    : {'thickness_um': 0.10, 'na_val': 1e18, 'defect': 1e14},
    }

    r = physics_precheck_detailed(valid, {}, mock_dev, apply_legacy_band_checks=False)
    if r['valid']:
        L_diff = r['details']['C1_diffusion_length'].get('L_diff_um', 0)
        print(f'  ✅ T1 PASS: Valid params accepted (L_diff={L_diff:.3f}μm)')
        passed += 1
    else:
        print(f'  ❌ T1 FAIL: {r["reason"]}')
        failed += 1

    degen = copy.deepcopy(valid)
    degen['CBTS']['na_val'] = 5e19
    r = physics_precheck_detailed(degen, {}, mock_dev, apply_legacy_band_checks=False)
    if not r['valid'] and 'C2_degenerate_doping' in r['failed_constraints']:
        print('  ✅ T2 PASS: Degenerate CBTS rejected')
        passed += 1
    else:
        print(f'  ❌ T2 FAIL: Expected C2, got {r["failed_constraints"]}')
        failed += 1

    bad_diff = copy.deepcopy(valid)
    bad_diff['MASnIBr2'] = {'thickness_um': 0.50, 'defect': 5e16}
    r = physics_precheck_detailed(bad_diff, {}, mock_dev, apply_legacy_band_checks=False)
    if not r['valid'] and 'C1_diffusion_length' in r['failed_constraints']:
        print('  ✅ T3 PASS: Diffusion length violation rejected')
        passed += 1
    else:
        print(f'  ❌ T3 FAIL: Expected C1, got {r["failed_constraints"]}')
        failed += 1

    too_thin = copy.deepcopy(valid)
    too_thin['MASnIBr2'] = {'thickness_um': 0.010, 'defect': 1e14}
    r = physics_precheck_detailed(too_thin, {}, mock_dev, apply_legacy_band_checks=False)
    if not r['valid'] and 'C4_thin_film_limit' in r['failed_constraints']:
        print('  ✅ T4 PASS: Thin-film limit rejected')
        passed += 1
    else:
        print(f'  ❌ T4 FAIL: Expected C4, got {r["failed_constraints"]}')
        failed += 1

    bad_junc = copy.deepcopy(valid)
    bad_junc['WO3']['nd_val'] = 1e14
    r = physics_precheck_detailed(bad_junc, {}, mock_dev, apply_legacy_band_checks=False)
    if not r['valid'] and 'C3_junction_formation' in r['failed_constraints']:
        print('  ✅ T5 PASS: Junction formation rejected')
        passed += 1
    else:
        print(f'  ❌ T5 FAIL: Expected C3, got {r["failed_constraints"]}')
        failed += 1

    baseline = {
        'WO3'     : {'thickness_um': 0.05, 'nd_val': 1e18, 'defect': 1e15},
        'MASnIBr2': {'thickness_um': 0.50, 'defect': 1e15},
        'CBTS'    : {'thickness_um': 0.10, 'na_val': 1e18, 'defect': 1e15},
    }
    r = physics_precheck_detailed(baseline, {}, mock_dev, apply_legacy_band_checks=False)
    if r['valid']:
        print('  ✅ T6 PASS: Baseline accepted')
        passed += 1
    else:
        print(f'  ❌ T6 FAIL: Baseline rejected — {r["reason"]}')
        failed += 1

    print(f'\n  Total: {passed}/6 passed')
    return failed == 0


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 10 — Boundary-inclusive sampler + objective function
# ============================================================
# EVAL_BOUNDARIES guarantees that every round evaluates
# at the min and max of each range, so the endpoints are never skipped.
# Implemented via optuna.trial.FixedTrial queued before TPE trials.
# ============================================================

def make_boundary_trials(current_ranges: dict, device: DeviceConfig) -> List[dict]:
    """Generate parameter dicts that pin parameters at their min or max boundaries.
    Returns a list of trial-param dicts to enqueue before TPE sampling starts.
    Strategy: for each parameter, create two trials — one at min, one at max.
    Other params set to geometric/arithmetic midpoint.
    """
    trials = []
    if not EVAL_BOUNDARIES:
        return trials

    def midpoint(lo, hi, log=False):
        if log:
            return 10 ** ((np.log10(lo) + np.log10(hi)) / 2)
        return (lo + hi) / 2

    # Build the "midpoint base trial"
    base = {}
    for l in device.layers:
        if not l.optimize: continue
        r = _active_param_ranges_for_role(l.role, current_ranges)
        if 'thickness' in r:
            base[f'{l.name}_t'] = midpoint(*r['thickness'], log=False)
        if 'defect' in r:
            base[f'{l.name}_defect'] = midpoint(*r['defect'], log=True)
        if 'NA' in r and l.doping_type in ('p', 'both'):
            base[f'{l.name}_NA'] = midpoint(*r['NA'], log=True)
        if 'ND' in r and l.doping_type in ('n', 'both'):
            base[f'{l.name}_ND'] = midpoint(*r['ND'], log=True)
    for key, (lo, hi) in device.interface_space.items():
        base[key] = midpoint(lo, hi, log=True)
    base['back_contact_wf'] = midpoint(*BACK_CONTACT_WF_RANGE, log=False)

    # For each parameter, create a boundary trial at min and max
    param_bounds = {}
    for l in device.layers:
        if not l.optimize: continue
        r = _active_param_ranges_for_role(l.role, current_ranges)
        if 'thickness' in r: param_bounds[f'{l.name}_t'] = r['thickness']
        if 'defect' in r:    param_bounds[f'{l.name}_defect'] = r['defect']
        if 'NA' in r and l.doping_type in ('p', 'both'): param_bounds[f'{l.name}_NA'] = r['NA']
        if 'ND' in r and l.doping_type in ('n', 'both'): param_bounds[f'{l.name}_ND'] = r['ND']
    for key, bounds in device.interface_space.items():
        param_bounds[key] = bounds
    param_bounds['back_contact_wf'] = BACK_CONTACT_WF_RANGE

    for param, (lo, hi) in param_bounds.items():
        for bound_val in [lo, hi]:
            t = dict(base)
            t[param] = bound_val
            trials.append(t)

    # Also add an "all-min" and "all-max" corner trial
    all_min = {p: lo for p, (lo, _) in param_bounds.items()}
    all_max = {p: hi for p, (_, hi) in param_bounds.items()}
    trials.append(all_min)
    trials.append(all_max)

    return trials


def make_objective(current_ranges: dict, round_num: int, seed: int,
                   device: DeviceConfig, runner: ScapsRunner, store: TrialStore,
                   template_file: str, destination_file: str,
                   script_file: str, iv_file: str, qe_file: str,
                   best_scaps_file: str):
    """Factory: returns an Optuna-compatible objective function."""

    def objective(trial):
        t0 = time.time()
        is_boundary = trial.user_attrs.get('is_boundary', 0)

        row = {
            'seed': seed,
            'round_number': round_num,
            'trial_number': trial.number,
            'is_boundary_eval': int(is_boundary),
            'parse_success': 0,
            'scaps_run_success': 0,
            'valid_run': 0,
            'wall_clock_start': t0,
        }

        def fail(reason: str) -> float:
            row['invalid_reason'] = reason
            row['runtime_seconds'] = round(time.time() - t0, 2)
            store.insert(row)
            return -100.0

        # Sample parameters
        layer_updates = {}
        for l in device.layers:
            if not l.optimize: continue
            r = _active_param_ranges_for_role(l.role, current_ranges)
            t = na_val = nd_val = defect = None

            if 'thickness' in r:
                t = trial.suggest_float(f'{l.name}_t', *r['thickness'])
                row[f'{l.name}_t'] = t
                row[f'{l.name}_thickness'] = t
            if 'defect' in r:
                defect = trial.suggest_float(f'{l.name}_defect', *r['defect'], log=True)
                row[f'{l.name}_defect'] = defect
            if 'NA' in r and l.doping_type in ('p', 'both'):
                na_val = trial.suggest_float(f'{l.name}_NA', *r['NA'], log=True)
                row[f'{l.name}_NA'] = na_val
            if 'ND' in r and l.doping_type in ('n', 'both'):
                nd_val = trial.suggest_float(f'{l.name}_ND', *r['ND'], log=True)
                row[f'{l.name}_ND'] = nd_val

            layer_updates[l.block_name] = {
                'thickness_um': t, 'doping_type': l.doping_type,
                'na_val': na_val, 'nd_val': nd_val, 'defect': defect
            }

        iface_vals = {}
        for key, (lo, hi) in device.interface_space.items():
            val = trial.suggest_float(key, lo, hi, log=True)
            iface_vals[key] = val
            row[key] = val

        # Back contact — optionally couple to HTL
        if COUPLE_WF_TO_HTL and device.get_layer_by_role('HTL'):
            htl = device.get_layer_by_role('HTL')
            if htl.name in MATERIAL_PARAMS:
                base_wf = MATERIAL_PARAMS[htl.name]['chi'] + MATERIAL_PARAMS[htl.name]['Eg']
                offset = trial.suggest_float('wf_offset', -0.1, 0.1)
                wf = base_wf + offset
            else:
                wf = trial.suggest_float('back_contact_wf', *BACK_CONTACT_WF_RANGE)
        else:
            wf = trial.suggest_float('back_contact_wf', *BACK_CONTACT_WF_RANGE)
        row['back_contact_wf'] = wf

        # Pre-check
        ok, reason = physics_precheck(layer_updates, iface_vals, device)
        if not ok:
            return fail(f'Pre-check: {reason}')

        # Build .scaps with verification
        try:
            build_updated_scaps_file(
                template_file, destination_file,
                layer_updates, iface_vals, back_contact_wf=wf, verify=True
            )
        except Exception as e:
            return fail(f'File build: {str(e)[:100]}')

        write_scaps_script(script_file, destination_file, iv_file, qe_file, run_qe=False)
        ok_run = runner.run(wait_for_qe=False)
        row['scaps_run_success'] = int(ok_run)
        if not ok_run:
            return fail('SCAPS timeout/fail')

        result = parse_iv(iv_file)
        if result is None:
            return fail('IV parse failed')

        Voc, Jsc, FF, eta = result
        row['parse_success'] = 1
        row.update({'Voc': Voc, 'Jsc': Jsc, 'FF': FF, 'eta': eta})

        valid, reason = physics_postcheck(Voc, Jsc, FF, eta, device)
        row.update({
            'valid_run': int(valid),
            'invalid_reason': reason,
            'runtime_seconds': round(time.time() - t0, 2)
        })
        store.insert(row)

        marker = '[B]' if is_boundary else '   '
        print(f'  {marker} S{seed} R{round_num}[{trial.number:>4d}] '
              f'Voc={Voc:.3f}V Jsc={Jsc:.2f} FF={FF:.1f}% PCE={eta:.3f}%  '
              f'{"OK" if valid else "INVALID: " + reason}')

        return eta if valid else -100.0

    return objective


print('Objective factory with boundary evaluation ready.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 11 — PPSR engine (SHAP + Optuna importance, ablation-aware)
# ============================================================

def compute_dual_importance(valid_df: pd.DataFrame, feature_cols: List[str],
                             optuna_study: Optional[optuna.Study] = None,
                             seed: int = 42) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (shap_importance, optuna_importance) — both normalized."""
    available = [c for c in feature_cols if c in valid_df.columns and pd.to_numeric(valid_df[c], errors='coerce').notna().any()]
    if len(valid_df) < 20 or not available:
        uniform = {c: 1.0/max(len(available), 1) for c in available}
        return uniform, uniform

    X = valid_df[available].fillna(valid_df[available].median())
    y = valid_df['eta']

    # SHAP via RandomForest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                                random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X)
    shap_imp = {c: float(np.abs(shap_vals[:, i]).mean())
                for i, c in enumerate(available)}
    total = sum(shap_imp.values())
    if total > 0:
        shap_imp = {k: v/total for k, v in shap_imp.items()}

    # Optuna native (fANOVA) — free, uses trial history directly
    optuna_imp = {}
    if optuna_study is not None and len(optuna_study.trials) > 10:
        try:
            raw = optuna.importance.get_param_importances(
                optuna_study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
            )
            # Map optuna param names back to feature_col names
            name_map = {}
            for c in available:
                if c.endswith('_thickness'):
                    name_map[c.replace('_thickness', '_t')] = c
                else:
                    name_map[c] = c
            for p, v in raw.items():
                if p in name_map:
                    optuna_imp[name_map[p]] = float(v)
            total = sum(optuna_imp.values())
            if total > 0:
                optuna_imp = {k: v/total for k, v in optuna_imp.items()}
        except Exception as e:
            print(f'  (fANOVA failed: {e} — using uniform optuna importance)')

    if not optuna_imp:
        optuna_imp = {c: 1.0/len(available) for c in available}

    shap_imp = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True))
    optuna_imp = dict(sorted(optuna_imp.items(), key=lambda x: x[1], reverse=True))
    return shap_imp, optuna_imp


def check_boundary_hit(valid_df: pd.DataFrame, current_ranges: dict,
                       feature_cols: List[str], device: DeviceConfig) -> Dict[str, bool]:
    top = valid_df.nlargest(max(10, int(len(valid_df)*0.2)), 'eta')
    hits = {}
    for role, ranges in current_ranges.items():
        active_ranges = _active_param_ranges_for_role(role, current_ranges)
        for param_type, (lo, hi) in active_ranges.items():
            col = _match_col(role, param_type, feature_cols, device)
            if col is None or col not in top.columns: continue
            vals = top[col].dropna()
            if len(vals) < 3: continue
            if vals.min() < lo * 1.05 or vals.max() > hi * 0.95:
                hits[col] = True
    return hits


def _match_col(role: str, param_type: str, feature_cols: List[str],
               device: DeviceConfig) -> Optional[str]:
    """Map (role, param_type) -> dataframe/feature column name.

    Handles the final optimizer naming convention where thickness is sampled as
    *_t, while older rows may still store *_thickness.
    """
    role_layer_names = [l.name for l in device.layers if l.role == role]
    candidates = []
    for ln in role_layer_names:
        if param_type == 'thickness':
            candidates.extend([f'{ln}_t', f'{ln}_thickness'])
        else:
            candidates.append(f'{ln}_{param_type}')
            candidates.append(f'{ln}_{param_type.upper()}')
            candidates.append(f'{ln}_{param_type.lower()}')

    for cand in candidates:
        if cand in feature_cols:
            return cand
    for fc in feature_cols:
        if any(ln.lower() in fc.lower() for ln in role_layer_names):
            if param_type == 'thickness' and (fc.endswith('_t') or fc.endswith('_thickness')):
                return fc
            if param_type.lower() in fc.lower():
                return fc
    return None


def auto_narrow_ranges(valid_df: pd.DataFrame, importance: dict,
                       current_ranges: dict, round_num: int,
                       feature_cols: List[str], device: DeviceConfig) -> dict:
    """PPSR core — narrow ranges based on top-performer distribution."""
    if not USE_PPSR_NARROWING:
        print('  (PPSR narrowing disabled — ablation mode)')
        return current_ranges

    top = valid_df.nlargest(max(10, int(len(valid_df)*0.2)), 'eta')
    new_ranges = {}
    boundary_hits = check_boundary_hit(valid_df, current_ranges, feature_cols, device)

    if boundary_hits:
        print(f'  ⚠ Boundary hits: {list(boundary_hits.keys())}')

    print(f'\n  Range update R{round_num}→R{round_num+1}:')
    print(f'  {"Parameter":<38} {"Imp":>6}  {"Action":<12} Old→New')
    print(f'  {"-"*85}')

    for role, ranges in current_ranges.items():
        new_layer = {}
        for param_type, (lo, hi) in _active_param_ranges_for_role(role, current_ranges).items():
            col = _match_col(role, param_type, feature_cols, device)

            if col is None or col not in top.columns:
                new_layer[param_type] = (lo, hi)
                continue

            imp = importance.get(col, 0)
            vals = top[col].dropna()

            if len(vals) < 5:
                new_layer[param_type] = (lo, hi)
                continue

            # Boundary → widen back toward initial
            if col in boundary_hits:
                init = INITIAL_PARAM_RANGES.get(role, {}).get(param_type, (lo, hi))
                new_lo = lo * 0.5 + init[0] * 0.5
                new_hi = hi * 0.5 + init[1] * 0.5
                new_layer[param_type] = (new_lo, new_hi)
                print(f'  {col:<38} {imp:>6.3f}  WIDEN        ({lo:.1e},{hi:.1e})→({new_lo:.1e},{new_hi:.1e})')
                continue

            is_log = (hi / max(lo, 1e-30)) > 100

            if imp > IMPORTANCE_THRESHOLD:
                if is_log:
                    lv = np.log10(vals.clip(lo, hi))
                    p_lo = np.percentile(lv, PERCENTILE_LO)
                    p_hi = np.percentile(lv, PERCENTILE_HI)
                    buf = (p_hi - p_lo) * BUFFER_FACTOR
                    new_lo = max(lo, 10**(p_lo - buf))
                    new_hi = min(hi, 10**(p_hi + buf))
                else:
                    p_lo = np.percentile(vals, PERCENTILE_LO)
                    p_hi = np.percentile(vals, PERCENTILE_HI)
                    buf = (p_hi - p_lo) * BUFFER_FACTOR
                    new_lo = max(lo, p_lo - buf)
                    new_hi = min(hi, p_hi + buf)

                if new_lo >= new_hi or new_hi/max(new_lo, 1e-30) < 1.1:
                    new_lo, new_hi = lo, hi
                    action = 'KEEP(tight)'
                else:
                    action = 'NARROW'
                new_layer[param_type] = (new_lo, new_hi)
                print(f'  {col:<38} {imp:>6.3f}  {action:<12} ({lo:.1e},{hi:.1e})→({new_lo:.1e},{new_hi:.1e})')

            elif imp > 0.01:
                if is_log:
                    lmid = (np.log10(lo) + np.log10(hi)) / 2
                    half = (np.log10(hi) - np.log10(lo)) * 0.35
                    new_lo = max(lo, 10**(lmid - half))
                    new_hi = min(hi, 10**(lmid + half))
                else:
                    mid = (lo + hi) / 2
                    half = (hi - lo) * 0.35
                    new_lo = max(lo, mid - half)
                    new_hi = min(hi, mid + half)
                new_layer[param_type] = (new_lo, new_hi)
                print(f'  {col:<38} {imp:>6.3f}  SLIGHT       ({lo:.1e},{hi:.1e})→({new_lo:.1e},{new_hi:.1e})')
            else:
                new_layer[param_type] = (lo, hi)
                print(f'  {col:<38} {imp:>6.3f}  KEEP         ({lo:.1e},{hi:.1e})')

        new_ranges[role] = new_layer
    return new_ranges


def check_convergence(best_history: List[float]) -> Tuple[bool, float]:
    if len(best_history) < CONVERGENCE_ROUNDS:
        return False, 0.0
    recent = best_history[-CONVERGENCE_ROUNDS:]
    improvement = max(recent) - min(recent)
    return improvement < CONVERGENCE_THRESHOLD, improvement


print('PPSR engine ready.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 12 — Joint-distribution range narrowing
# ============================================================
# Replaces per-parameter percentile narrowing with a multivariate Gaussian fit
# on top-K trials. This captures parameter correlations (e.g., thickness-doping
# coupling) that marginal narrowing destroys.
#
# Falls back to marginal narrowing if:
#   - Fewer than K valid trials (not enough data to fit covariance)
#   - Singular covariance (some parameter has zero variance in top-K)
#   - User disables with USE_JOINT_NARROWING=False
# ============================================================

def _map_feature_to_role_param(feature_col: str, device: DeviceConfig) -> Optional[Tuple[str, str]]:
    """Given a feature column like 'MASnIBr2_NA', return ('ABSORBER', 'NA'). None if no match."""
    for l in device.layers:
        if not l.optimize:
            continue
        if feature_col.startswith(l.name + '_'):
            param_type = feature_col[len(l.name) + 1:]
            return (l.role, param_type)
    return None


def joint_narrow_ranges(valid_df: pd.DataFrame, current_ranges: dict,
                         round_num: int, device: DeviceConfig,
                         top_k: int = 20, sigma_factor: float = 2.0,
                         importance: Optional[dict] = None) -> dict:
    """Narrow ranges using multivariate Gaussian on top-K trials.

    Strategy:
      1. Select top-K trials by PCE
      2. Log-transform columns that span orders of magnitude
      3. Fit multivariate mean and covariance
      4. For each parameter: new_range = mean ± sigma_factor * sqrt(diag(cov))
         BUT respect the original range as hard bounds
      5. If parameter importance is low, keep wider range (same as marginal method)
    """
    if not USE_JOINT_NARROWING:
        return None  # signal fallback to marginal

    available = [c for c in device.feature_cols if c in valid_df.columns]
    if len(valid_df) < top_k or not available:
        print(f'  (joint narrowing: only {len(valid_df)} valid trials, need {top_k} — fallback to marginal)')
        return None

    top = valid_df.nlargest(top_k, 'eta')

    # Log-transform columns that span > 2 orders of magnitude
    X = top[available].copy()
    log_cols = []
    for c in X.columns:
        col_vals = X[c].dropna()
        if len(col_vals) < 3:
            continue
        if col_vals.min() > 0 and col_vals.max() / col_vals.min() > 100:
            X[c] = np.log10(col_vals)
            log_cols.append(c)

    X = X.fillna(X.median())

    mean = X.mean().values
    try:
        cov = X.cov().values
        # Regularize for numerical stability
        cov = cov + np.eye(len(cov)) * 1e-12
    except Exception as e:
        print(f'  (joint narrowing: covariance failed ({e}) — fallback to marginal)')
        return None

    std = np.sqrt(np.diag(cov))

    # Check for near-singular (zero-variance) directions
    if np.any(std < 1e-15):
        print(f'  (joint narrowing: some parameter has ~zero variance — fallback to marginal)')
        return None

    # Build new ranges
    new_ranges = {}
    narrow_log = []

    for role, ranges in current_ranges.items():
        new_layer = {}
        for param_type, (lo, hi) in _active_param_ranges_for_role(role, current_ranges).items():
            col = _match_col(role, param_type, device.feature_cols, device)

            if col is None or col not in X.columns:
                new_layer[param_type] = (lo, hi)
                continue

            idx = list(X.columns).index(col)
            mu = mean[idx]
            sd = std[idx]

            # Check importance — if low, keep wider (same idea as marginal method)
            imp = importance.get(col, 0) if importance else IMPORTANCE_THRESHOLD + 0.01

            if imp < 0.01:
                # Unimportant — keep original
                new_layer[param_type] = (lo, hi)
                narrow_log.append((col, imp, 'KEEP', lo, hi, lo, hi))
                continue

            # Compute joint-narrowed range
            if col in log_cols:
                # In log space
                log_lo = mu - sigma_factor * sd
                log_hi = mu + sigma_factor * sd
                new_lo = max(lo, 10 ** log_lo)
                new_hi = min(hi, 10 ** log_hi)
            else:
                new_lo = max(lo, mu - sigma_factor * sd)
                new_hi = min(hi, mu + sigma_factor * sd)

            # Degenerate check
            if new_lo >= new_hi or new_hi / max(new_lo, 1e-30) < 1.1:
                new_lo, new_hi = lo, hi
                action = 'KEEP(tight)'
            else:
                action = 'JOINT-NARROW' if imp > IMPORTANCE_THRESHOLD else 'JOINT-SLIGHT'

            new_layer[param_type] = (new_lo, new_hi)
            narrow_log.append((col, imp, action, lo, hi, new_lo, new_hi))

        new_ranges[role] = new_layer

    # Print report
    print(f'\n  Joint-distribution range update R{round_num}→R{round_num+1}:')
    print(f'  {"Parameter":<38} {"Imp":>6}  {"Action":<14} Old→New')
    print(f'  {"-"*95}')
    for col, imp, action, lo, hi, nlo, nhi in narrow_log:
        print(f'  {col:<38} {imp:>6.3f}  {action:<14} ({lo:.1e},{hi:.1e})→({nlo:.1e},{nhi:.1e})')

    # Also print cross-correlations (paper-worthy)
    print(f'\n  Top parameter correlations in top-{top_k} (|r| > 0.5):')
    corr = X.corr()
    pairs_shown = 0
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f'    {corr.columns[i]:<25} ↔ {corr.columns[j]:<25} r = {r:+.3f}')
                pairs_shown += 1
    if pairs_shown == 0:
        print('    (no |r| > 0.5 pairs)')

    return new_ranges


# auto_narrow_ranges tries joint narrowing first
_original_auto_narrow = auto_narrow_ranges

def auto_narrow_ranges_v3(valid_df, importance, current_ranges, round_num,
                           feature_cols, device):
    """v3: try joint narrowing first, fall back to marginal if unavailable."""
    if not USE_PPSR_NARROWING:
        print('  (PPSR narrowing disabled — ablation mode)')
        return current_ranges

    if USE_JOINT_NARROWING:
        joint_result = joint_narrow_ranges(
            valid_df, current_ranges, round_num, device,
            top_k=JOINT_NARROW_TOP_K,
            sigma_factor=JOINT_SIGMA_FACTOR,
            importance=importance,
        )
        if joint_result is not None:
            return joint_result

    # Fallback to marginal narrowing
    return _original_auto_narrow(valid_df, importance, current_ranges, round_num,
                                  feature_cols, device)


# Replace the global reference
auto_narrow_ranges = auto_narrow_ranges_v3
print('Joint-distribution narrowing ready (v3).')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 13 — Statistically correct bootstrap + multi-seed aggregation
# ============================================================
# Bootstrap mean of top-K performers (representative of the optimal region),
# AND aggregate across N_SEEDS independent runs for true reproducibility CI.
# ============================================================

def bootstrap_top_k_mean(valid_df: pd.DataFrame, k: int = 10,
                         n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI on mean-of-top-K PCE.
    This measures the stability of the *optimal region*, not just one point.
    """
    pce = valid_df['eta'].values
    if len(pce) < k:
        k = max(1, len(pce)//2)

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        resample = rng.choice(pce, size=len(pce), replace=True)
        top_k = np.sort(resample)[-k:]
        boot_means.append(top_k.mean())
    boot_means = np.array(boot_means)

    observed_top_k_mean = np.sort(pce)[-k:].mean()
    observed_max = pce.max()
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    return {
        'observed_max': float(observed_max),
        'observed_top_k_mean': float(observed_top_k_mean),
        'boot_mean': float(boot_means.mean()),
        'boot_std': float(boot_means.std()),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'k': k,
        'n_boot': n_boot,
        'boot_samples': boot_means,
    }


def multi_seed_statistics(per_seed_champions: List[float]) -> dict:
    """Statistics over independent PPSR runs. This is the REAL confidence interval."""
    arr = np.array(per_seed_champions)
    n = len(arr)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    # t-based 95% CI
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_lo = mean - t_crit * sem
        ci_hi = mean + t_crit * sem
    else:
        ci_lo = ci_hi = mean
    return {
        'n_seeds': n,
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'ci_95_lo': float(ci_lo),
        'ci_95_hi': float(ci_hi),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'all_values': arr.tolist(),
    }


def plot_bootstrap(stats_dict: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(stats_dict['boot_samples'], bins=60, color='#1565C0',
            alpha=0.7, edgecolor='white')
    ax.axvline(stats_dict['observed_max'], color='red', lw=2.2,
               label=f"Observed max = {stats_dict['observed_max']:.3f}%")
    ax.axvline(stats_dict['observed_top_k_mean'], color='darkgreen', lw=2.2,
               label=f"Top-{stats_dict['k']} mean = {stats_dict['observed_top_k_mean']:.3f}%")
    ax.axvline(stats_dict['ci_lo'], color='orange', lw=1.5, ls='--',
               label=f"95% CI: [{stats_dict['ci_lo']:.3f}, {stats_dict['ci_hi']:.3f}]%")
    ax.axvline(stats_dict['ci_hi'], color='orange', lw=1.5, ls='--')
    ax.set_xlabel('PCE (%)')
    ax.set_ylabel('Frequency (bootstrap resamples)')
    ax.set_title(f"Bootstrap CI on Top-{stats_dict['k']} Mean PCE  (n={stats_dict['n_boot']})",
                 fontweight='bold')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


print('Proper bootstrap + multi-seed statistics ready.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 15 — Main experiment: multi-seed PPSR with full validation
# ============================================================
# Runs N_SEEDS independent PPSR optimizations and aggregates statistics.
# Supports SQLite checkpoint resume — safe to interrupt and restart.
# ============================================================

DB_PATH = os.path.join(OUT_DIR, 'trials.db')
OPTUNA_DB = f'sqlite:///{os.path.join(OUT_DIR, "optuna_study.db")}'

def run_single_seed(seed: int, device: DeviceConfig, runner: ScapsRunner,
                    store: TrialStore) -> dict:
    """One complete PPSR optimization at a given seed."""
    random.seed(seed)
    np.random.seed(seed)

    print(f'\n{"═"*70}')
    print(f'   SEED {seed} — PPSR run')
    print(f'{"═"*70}')

    current_ranges = {k: dict(v) for k, v in INITIAL_PARAM_RANGES.items()}
    previous_ranges = None
    global_best = 0.0
    best_per_round = []
    importance_history = []
    range_history = [{'round': 0, 'ranges': current_ranges}]
    wall_start = time.time()

    best_scaps_file = os.path.join(OUT_DIR, f'best_device_seed{seed}.scaps')

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f'\n{"-"*70}')
        print(f'Seed {seed} — Round {round_num}/{MAX_ROUNDS}  |  Global best: {global_best:.4f}%')
        print(f'{"-"*70}')

        sampler = optuna.samplers.TPESampler(seed=seed * 1000 + round_num)
        study_name = f'seed{seed}_round{round_num}'

        try:
            study = optuna.create_study(
                direction='maximize', sampler=sampler,
                study_name=study_name, storage=OPTUNA_DB,
                load_if_exists=True
            )
        except Exception:
            study = optuna.create_study(direction='maximize', sampler=sampler,
                                         study_name=study_name)

        # Enqueue boundary trials BEFORE TPE starts
        boundary_trials = make_boundary_trials(current_ranges, device)
        for bt in boundary_trials:
            study.enqueue_trial(bt, user_attrs={'is_boundary': 1})
        if boundary_trials:
            print(f'  Enqueued {len(boundary_trials)} boundary trials (min/max endpoints)')

        n_trials = TRIALS_PER_ROUND + len(boundary_trials) if EVAL_BOUNDARIES else TRIALS_PER_ROUND

        objective = make_objective(
            current_ranges, round_num, seed, device, runner, store,
            TEMPLATE_FILE, DESTINATION_FILE, SCRIPT_FILE,
            OUTPUT_IV_FILE, OUTPUT_QE_FILE, best_scaps_file
        )
        study.optimize(objective, n_trials=n_trials)

        # Read back from DB
        valid_df = store.valid_df(seed=seed)
        round_df = valid_df[valid_df['round_number'] == round_num]
        round_best = round_df['eta'].max() if len(round_df) > 0 else 0.0

        print(f'\n  Round {round_num} best: {round_best:.4f}%')
        print(f'  Round {round_num} valid: {len(round_df)}/{n_trials} trials')

        if len(valid_df) == 0:
            best_per_round.append(global_best)
            continue

        # Rollback
        if USE_ROLLBACK and global_best > 0 and round_best < global_best * ROLLBACK_THRESHOLD:
            print(f'  ⚠ ROLLBACK: {round_best:.3f}% < {global_best*ROLLBACK_THRESHOLD:.3f}%')
            if previous_ranges:
                current_ranges = previous_ranges
            best_per_round.append(global_best)
            continue

        if round_best > global_best:
            global_best = round_best
            best_row = valid_df.nlargest(1, 'eta').iloc[0]
            save_best_device_scaps(best_row, current_ranges, device, best_scaps_file)
            print(f'  ★ NEW GLOBAL BEST (seed {seed}): {global_best:.4f}%')

        best_per_round.append(global_best)

        shap_imp, optuna_imp = compute_dual_importance(
            valid_df, device.feature_cols, study, seed=seed)
        importance_history.append({'shap': shap_imp, 'optuna': optuna_imp})

        print(f'  Top 5 SHAP    : ' + ', '.join(f'{k}={v:.2f}' for k, v in list(shap_imp.items())[:5]))
        print(f'  Top 5 fANOVA  : ' + ', '.join(f'{k}={v:.2f}' for k, v in list(optuna_imp.items())[:5]))

        converged, improvement = check_convergence(best_per_round)
        print(f'  Convergence: improvement={improvement:.4f}% (threshold={CONVERGENCE_THRESHOLD}%)')

        if converged:
            print(f'  ✓ CONVERGED at round {round_num}')
            break

        if round_num < MAX_ROUNDS:
            previous_ranges = {k: dict(v) for k, v in current_ranges.items()}
            current_ranges = auto_narrow_ranges(
                valid_df, shap_imp, current_ranges, round_num,
                device.feature_cols, device)
            range_history.append({'round': round_num, 'ranges': current_ranges})

    wall_time = time.time() - wall_start
    return {
        'seed': seed,
        'global_best': global_best,
        'best_per_round': best_per_round,
        'importance_history': importance_history,
        'range_history': range_history,
        'wall_time_seconds': wall_time,
        'total_trials': store.count(seed=seed),
        'valid_trials': len(store.valid_df(seed=seed)),
    }


def _row_get_first(row, *keys, default=None):
    for k in keys:
        try:
            v = row.get(k)
        except Exception:
            v = None
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and pd.notna(v):
            return v
    return default


def save_best_device_scaps(best_row, current_ranges, device, out_file):
    """Write optimal device to .scaps for sweeps/post-analysis."""
    layer_updates = {}
    for l in device.layers:
        if not l.optimize:
            continue
        r = current_ranges.get(l.role, {})
        t = _row_get_first(best_row, f'{l.name}_t', f'{l.name}_thickness') if 'thickness' in r else None
        defect = _row_get_first(best_row, f'{l.name}_defect') if 'defect' in r else None
        na_val = _row_get_first(best_row, f'{l.name}_NA') if ('NA' in r and l.doping_type in ('p','both')) else None
        nd_val = _row_get_first(best_row, f'{l.name}_ND') if ('ND' in r and l.doping_type in ('n','both')) else None
        layer_updates[l.block_name] = {
            'thickness_um': t,
            'doping_type': l.doping_type,
            'na_val': na_val,
            'nd_val': nd_val,
            'defect': defect,
        }

    iface_vals = {}
    for key in device.interface_space:
        v = _row_get_first(best_row, key)
        if v is not None:
            iface_vals[key] = v
        else:
            lo, hi = INTERFACE_RANGE
            iface_vals[key] = np.sqrt(lo * hi)

    back_wf = _row_get_first(best_row, 'back_contact_wf', default=VALIDATED_BACK_WF)
    build_updated_scaps_file(
        TEMPLATE_FILE, out_file, layer_updates, iface_vals,
        back_contact_wf=back_wf, verify=False,
    )


def run_experiment() -> dict:
    """Full multi-seed PPSR experiment. This is the main entry point."""
    print('=' * 70)
    print('PPSR v2.0 — MULTI-SEED EXPERIMENT')
    print('=' * 70)
    print(f'  Seeds             : {SEEDS}')
    print(f'  Trials per round  : {TRIALS_PER_ROUND}')
    print(f'  Max rounds        : {MAX_ROUNDS}')
    print(f'  Boundary eval     : {EVAL_BOUNDARIES}')
    print(f'  Ablation: PPSR={USE_PPSR_NARROWING} PRE={USE_PHYSICS_PRECHECK} '
          f'POST={USE_PHYSICS_POSTCHECK} RB={USE_ROLLBACK}')
    print('=' * 70)

    store = TrialStore(DB_PATH, device.feature_cols)
    runner = ScapsRunner(SCAPS_EXE, WORKING_DIR, SCRIPT_FILE,
                          OUTPUT_IV_FILE, OUTPUT_QE_FILE, timeout=MAX_TIMEOUT)

    per_seed_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, device, runner, store)
        per_seed_results.append(result)

    # Aggregate
    champions = [r['global_best'] for r in per_seed_results]
    multi_seed = multi_seed_statistics(champions)

    print('\n' + '═'*70)
    print('MULTI-SEED SUMMARY')
    print('═'*70)
    print(f'  Champions per seed: {[f"{c:.3f}" for c in champions]}')
    print(f'  Mean ± std        : {multi_seed["mean"]:.4f} ± {multi_seed["std"]:.4f}%')
    print(f'  95% CI (t-dist)   : [{multi_seed["ci_95_lo"]:.4f}, {multi_seed["ci_95_hi"]:.4f}]%')
    print(f'  Range             : [{multi_seed["min"]:.4f}, {multi_seed["max"]:.4f}]%')
    total_wall = sum(r['wall_time_seconds'] for r in per_seed_results)
    total_trials = sum(r['total_trials'] for r in per_seed_results)
    print(f'  Total wall-time   : {total_wall/3600:.2f} hours')
    print(f'  Total trials      : {total_trials}')
    print(f'  Avg sec/trial     : {total_wall/max(total_trials,1):.2f}')

    # Bootstrap on aggregated dataset
    all_valid = store.valid_df()
    boot = bootstrap_top_k_mean(all_valid, k=BOOTSTRAP_TOP_K, n_boot=BOOTSTRAP_N)
    print(f'\n  Bootstrap top-{boot["k"]} mean: {boot["observed_top_k_mean"]:.4f}%')
    print(f'  Bootstrap 95% CI     : [{boot["ci_lo"]:.4f}, {boot["ci_hi"]:.4f}]%')
    plot_bootstrap(boot, os.path.join(OUT_DIR, 'bootstrap_topK.png'))

    # Save summary JSON
    summary = {
        'config': {
            'n_seeds': N_SEEDS,
            'trials_per_round': TRIALS_PER_ROUND,
            'max_rounds': MAX_ROUNDS,
            'eval_boundaries': EVAL_BOUNDARIES,
            'ablation': {
                'ppsr_narrowing': USE_PPSR_NARROWING,
                'physics_precheck': USE_PHYSICS_PRECHECK,
                'physics_postcheck': USE_PHYSICS_POSTCHECK,
                'rollback': USE_ROLLBACK,
            },
        },
        'multi_seed': multi_seed,
        'bootstrap': {k: v for k, v in boot.items() if k != 'boot_samples'},
        'per_seed': [
            {
                'seed': r['seed'],
                'global_best': r['global_best'],
                'best_per_round': r['best_per_round'],
                'wall_time_seconds': r['wall_time_seconds'],
                'total_trials': r['total_trials'],
                'valid_trials': r['valid_trials'],
            } for r in per_seed_results
        ],
        'total_wall_hours': total_wall / 3600,
        'total_trials': total_trials,
    }
    with open(os.path.join(OUT_DIR, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  Summary saved: {os.path.join(OUT_DIR, "experiment_summary.json")}')

    return {
        'per_seed_results': per_seed_results,
        'multi_seed': multi_seed,
        'bootstrap': boot,
        'store': store,
    }


print('Main experiment runner ready.')
print('Call: results = run_experiment()')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 22 — Vanilla BO baseline experiment
# ============================================================
# Same trial budget as PPSR, but no narrowing. Multi-seed.
# This is the KEY comparison for Paper 1.
# ============================================================

BASELINE_DB_PATH = os.path.join(OUT_DIR, 'baseline_trials.db')
BASELINE_OPTUNA_DB = f'sqlite:///{os.path.join(OUT_DIR, "baseline_optuna.db")}'


def run_baseline_single_seed(seed: int, total_trials: int,
                              device: DeviceConfig, runner: ScapsRunner,
                              store: TrialStore) -> dict:
    """Run ONE vanilla BO experiment (no narrowing, single round, all trials)."""
    random.seed(seed)
    np.random.seed(seed)

    print(f'\n{"═"*70}')
    print(f'   BASELINE SEED {seed} — vanilla BO ({total_trials} trials)')
    print(f'{"═"*70}')

    wall_start = time.time()

    # Use INITIAL_PARAM_RANGES always — never narrow
    ranges = {k: dict(v) for k, v in INITIAL_PARAM_RANGES.items()}

    sampler = optuna.samplers.TPESampler(seed=seed * 1000 + 1)
    study_name = f'baseline_seed{seed}'
    try:
        study = optuna.create_study(
            direction='maximize', sampler=sampler,
            study_name=study_name, storage=BASELINE_OPTUNA_DB,
            load_if_exists=True,
        )
    except Exception:
        study = optuna.create_study(direction='maximize', sampler=sampler,
                                     study_name=study_name)

    # Optional boundary trials at start (same as PPSR, for fairness)
    if EVAL_BOUNDARIES:
        boundary_trials = make_boundary_trials(ranges, device)
        for bt in boundary_trials:
            study.enqueue_trial(bt, user_attrs={'is_boundary': 1})
        total_with_boundary = total_trials + len(boundary_trials)
        print(f'  Enqueued {len(boundary_trials)} boundary trials')
    else:
        total_with_boundary = total_trials

    best_scaps_file = os.path.join(OUT_DIR, f'baseline_best_seed{seed}.scaps')

    # round_num is always 1 for baseline
    objective = make_objective(
        ranges, round_num=1, seed=seed, device=device,
        runner=runner, store=store,
        template_file=TEMPLATE_FILE, destination_file=DESTINATION_FILE,
        script_file=SCRIPT_FILE, iv_file=OUTPUT_IV_FILE, qe_file=OUTPUT_QE_FILE,
        best_scaps_file=best_scaps_file,
    )

    study.optimize(objective, n_trials=total_with_boundary)

    valid_df = store.valid_df(seed=seed)
    global_best = valid_df['eta'].max() if len(valid_df) else 0.0

    if len(valid_df) > 0:
        best_row = valid_df.nlargest(1, 'eta').iloc[0]
        save_best_device_scaps(best_row, ranges, device, best_scaps_file)

    wall_time = time.time() - wall_start

    return {
        'seed': seed,
        'global_best': global_best,
        'wall_time_seconds': wall_time,
        'total_trials': store.count(seed=seed),
        'valid_trials': len(valid_df),
    }


def run_baseline_experiment() -> dict:
    """Run vanilla BO baseline for all seeds, matching PPSR trial budget."""
    if not RUN_BASELINE:
        print('RUN_BASELINE = False — skipping baseline experiment')
        return None

    # Auto-match trial budget to PPSR
    if BASELINE_TOTAL_TRIALS is None:
        # PPSR uses TRIALS_PER_ROUND per round, up to MAX_ROUNDS rounds,
        # typically converges at ~3-5 rounds. Match the AVERAGE.
        avg_rounds = (MAX_ROUNDS + CONVERGENCE_ROUNDS) / 2
        matched_trials = int(TRIALS_PER_ROUND * avg_rounds)
    else:
        matched_trials = BASELINE_TOTAL_TRIALS

    print('=' * 70)
    print('v3: VANILLA BO BASELINE EXPERIMENT')
    print('=' * 70)
    print(f'  Seeds             : {SEEDS[:N_SEEDS]}')
    print(f'  Trials per seed   : {matched_trials} (+ boundaries if enabled)')
    print(f'  No narrowing, no rounds, no SHAP-guidance')
    print('=' * 70)

    store = TrialStore(BASELINE_DB_PATH, device.feature_cols)
    runner = ScapsRunner(SCAPS_EXE, WORKING_DIR, SCRIPT_FILE,
                          OUTPUT_IV_FILE, OUTPUT_QE_FILE, timeout=MAX_TIMEOUT)

    per_seed = []
    for seed in SEEDS[:N_SEEDS]:
        r = run_baseline_single_seed(seed, matched_trials, device, runner, store)
        per_seed.append(r)

    champions = [r['global_best'] for r in per_seed]
    ms = multi_seed_statistics(champions)

    print('\n' + '═'*70)
    print('BASELINE SUMMARY')
    print('═'*70)
    print(f'  Champions per seed: {[f"{c:.3f}" for c in champions]}')
    print(f'  Mean ± std        : {ms["mean"]:.4f} ± {ms["std"]:.4f}%')
    print(f'  95% CI            : [{ms["ci_95_lo"]:.4f}, {ms["ci_95_hi"]:.4f}]%')
    total_wall = sum(r['wall_time_seconds'] for r in per_seed)
    print(f'  Total wall time   : {total_wall/3600:.2f} hours')

    summary = {
        'per_seed': per_seed,
        'multi_seed': ms,
        'store': store,
        'total_wall_seconds': total_wall,
        'matched_trials_per_seed': matched_trials,
    }

    with open(os.path.join(OUT_DIR, 'baseline_summary.json'), 'w') as f:
        json.dump({
            'multi_seed': ms,
            'per_seed': [{k: v for k, v in r.items()} for r in per_seed],
            'total_wall_hours': total_wall / 3600,
            'matched_trials_per_seed': matched_trials,
        }, f, indent=2)

    return summary


print('Baseline (TPE) runner defined.')


# ####################################################################################
# From the original notebook
# ####################################################################################

# CELL 23 — Genetic Algorithm baseline experiment
# ============================================================
# Uses the SAME SCAPS objective, precheck, postcheck, and logging as PPSR/BO.
# GA is added as a separate optimizer benchmark, not a replacement for PPSR.
# ============================================================

GA_DB_PATH = os.path.join(OUT_DIR, 'ga_trials.db')

class FixedParamTrial:
    """Minimal Optuna-like trial wrapper so GA can reuse make_objective()."""
    def __init__(self, params: dict, number: int, user_attrs: Optional[dict] = None):
        self.params = dict(params)
        self.number = int(number)
        self.user_attrs = user_attrs or {}

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if name not in self.params:
            val = np.sqrt(low * high) if log else (low + high) / 2
            self.params[name] = float(val)
        val = float(self.params[name])
        if not np.isfinite(val):
            raise ValueError(f'GA param {name} is not finite')
        if log and val <= 0:
            raise ValueError(f'GA param {name} must be > 0 for log sampling')
        if val < low or val > high:
            val = min(max(val, low), high)
            self.params[name] = float(val)
        return float(val)


def ga_search_space(device: DeviceConfig, current_ranges: Optional[dict] = None) -> Dict[str, Tuple[float, float, bool]]:
    """Return flat param space: key -> (lo, hi, log_flag)."""
    ranges = current_ranges or INITIAL_PARAM_RANGES
    space = {}
    for l in device.layers:
        if not l.optimize:
            continue
        r = _active_param_ranges_for_role(l.role, ranges)
        if 'thickness' in r:
            space[f'{l.name}_t'] = (*r['thickness'], False)
        if 'defect' in r:
            space[f'{l.name}_defect'] = (*r['defect'], True)
        if 'NA' in r and l.doping_type in ('p', 'both'):
            space[f'{l.name}_NA'] = (*r['NA'], True)
        if 'ND' in r and l.doping_type in ('n', 'both'):
            space[f'{l.name}_ND'] = (*r['ND'], True)
    if OPTIMIZE_INTERFACE_DEFECTS:
        for key, (lo, hi) in device.interface_space.items():
            space[key] = (lo, hi, True)
    space['back_contact_wf'] = (*BACK_CONTACT_WF_RANGE, False)
    return space


def _ga_random_value(lo: float, hi: float, log: bool, rng) -> float:
    if log:
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))


def _ga_mutate_value(val: float, lo: float, hi: float, log: bool, rng, sigma: float = 0.15) -> float:
    if log:
        x = np.log10(max(val, 1e-300))
        lo_x, hi_x = np.log10(lo), np.log10(hi)
        x = x + rng.normal(0.0, sigma * (hi_x - lo_x))
        x = min(max(x, lo_x), hi_x)
        return float(10 ** x)
    span = hi - lo
    x = val + rng.normal(0.0, sigma * span)
    x = min(max(x, lo), hi)
    return float(x)


def _ga_random_individual(space: dict, rng) -> dict:
    return {k: _ga_random_value(lo, hi, log, rng) for k, (lo, hi, log) in space.items()}


def _ga_tournament(population: List[dict], fitnesses: List[float], k: int, rng) -> dict:
    idxs = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return dict(population[best_idx])


def _ga_crossover(parent_a: dict, parent_b: dict, space: dict, rng, crossover_rate: float = 0.8) -> Tuple[dict, dict]:
    if rng.random() > crossover_rate:
        return dict(parent_a), dict(parent_b)

    child_a, child_b = {}, {}
    for key, (lo, hi, log) in space.items():
        a = float(parent_a[key]); b = float(parent_b[key])
        alpha = rng.random()
        if log:
            xa = np.log10(max(a, 1e-300)); xb = np.log10(max(b, 1e-300))
            ya = alpha * xa + (1 - alpha) * xb
            yb = alpha * xb + (1 - alpha) * xa
            lo_x, hi_x = np.log10(lo), np.log10(hi)
            ya = min(max(ya, lo_x), hi_x)
            yb = min(max(yb, lo_x), hi_x)
            child_a[key] = float(10 ** ya)
            child_b[key] = float(10 ** yb)
        else:
            ya = alpha * a + (1 - alpha) * b
            yb = alpha * b + (1 - alpha) * a
            child_a[key] = float(min(max(ya, lo), hi))
            child_b[key] = float(min(max(yb, lo), hi))
    return child_a, child_b


def _ga_mutate(individual: dict, space: dict, rng, mutation_rate: float = 0.15, sigma: float = 0.15) -> dict:
    out = dict(individual)
    for key, (lo, hi, log) in space.items():
        if rng.random() < mutation_rate:
            out[key] = _ga_mutate_value(out[key], lo, hi, log, rng, sigma=sigma)
    return out


def run_ga_single_seed(seed: int, total_trials: int,
                       device: DeviceConfig, runner: ScapsRunner,
                       store: TrialStore) -> dict:
    """Run one GA benchmark with the same objective used by PPSR/BO."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed * 3001 + 17)

    print(f'\n{"═"*70}')
    print(f'   GA SEED {seed} — population={GA_POP_SIZE}, budget={total_trials}')
    print(f'{"═"*70}')

    wall_start = time.time()
    ranges = {k: dict(v) for k, v in INITIAL_PARAM_RANGES.items()}
    space = ga_search_space(device, ranges)
    best_scaps_file = os.path.join(OUT_DIR, f'ga_best_seed{seed}.scaps')

    population = []
    if EVAL_BOUNDARIES:
        for bt in make_boundary_trials(ranges, device):
            individual = {k: float(bt[k]) for k in space if k in bt}
            if len(individual) == len(space):
                population.append(individual)

    while len(population) < GA_POP_SIZE:
        population.append(_ga_random_individual(space, rng))
    population = population[:GA_POP_SIZE]

    total_budget = total_trials + (len(make_boundary_trials(ranges, device)) if EVAL_BOUNDARIES else 0)
    eval_count = 0
    trial_number = 0
    best_per_generation = []

    def evaluate_population(population_batch: List[dict], generation: int) -> List[float]:
        nonlocal eval_count, trial_number
        objective = make_objective(
            ranges, round_num=generation, seed=seed, device=device,
            runner=runner, store=store,
            template_file=TEMPLATE_FILE, destination_file=DESTINATION_FILE,
            script_file=SCRIPT_FILE, iv_file=OUTPUT_IV_FILE, qe_file=OUTPUT_QE_FILE,
            best_scaps_file=best_scaps_file,
        )
        fitnesses = []
        for i, individual in enumerate(population_batch):
            if eval_count >= total_budget:
                break
            user_attrs = {'is_boundary': int(EVAL_BOUNDARIES and generation == 0 and i < len(make_boundary_trials(ranges, device)))}
            trial = FixedParamTrial(individual, number=trial_number, user_attrs=user_attrs)
            fit = float(objective(trial))
            fitnesses.append(fit)
            eval_count += 1
            trial_number += 1
        return fitnesses

    fitnesses = evaluate_population(population, generation=0)
    generation = 0

    while True:
        valid_fits = fitnesses if len(fitnesses) == len(population) else fitnesses + [-100.0] * (len(population) - len(fitnesses))
        order = np.argsort(valid_fits)[::-1]
        population = [population[i] for i in order]
        valid_fits = [valid_fits[i] for i in order]

        current_best = max(valid_fits) if valid_fits else -100.0
        best_per_generation.append(current_best)
        print(f'  Generation {generation:>3d}: best={current_best:.4f}%  evals={eval_count}/{total_budget}')

        if eval_count >= total_budget:
            break

        elites_n = min(GA_ELITISM, len(population))
        new_population = [dict(population[i]) for i in range(elites_n)]
        new_fitnesses = [float(valid_fits[i]) for i in range(elites_n)]

        while len(new_population) < GA_POP_SIZE and eval_count < total_budget:
            parent_a = _ga_tournament(population, valid_fits, GA_TOURNAMENT_K, rng)
            parent_b = _ga_tournament(population, valid_fits, GA_TOURNAMENT_K, rng)
            child_a, child_b = _ga_crossover(parent_a, parent_b, space, rng, crossover_rate=GA_CROSSOVER_RATE)
            child_a = _ga_mutate(child_a, space, rng, mutation_rate=GA_MUTATION_RATE, sigma=GA_MUTATION_SIGMA)
            child_b = _ga_mutate(child_b, space, rng, mutation_rate=GA_MUTATION_RATE, sigma=GA_MUTATION_SIGMA)

            children = [child_a, child_b]
            child_fits = evaluate_population(children, generation=generation + 1)
            for child, fit in zip(children, child_fits):
                if len(new_population) >= GA_POP_SIZE:
                    break
                new_population.append(child)
                new_fitnesses.append(float(fit))
            if eval_count >= total_budget:
                break

        population, fitnesses = new_population, new_fitnesses
        generation += 1

    valid_df = store.valid_df(seed=seed)
    global_best = valid_df['eta'].max() if len(valid_df) else 0.0
    if len(valid_df) > 0:
        best_row = valid_df.nlargest(1, 'eta').iloc[0]
        save_best_device_scaps(best_row, ranges, device, best_scaps_file)

    wall_time = time.time() - wall_start
    return {
        'seed': seed,
        'global_best': global_best,
        'best_per_generation': best_per_generation,
        'wall_time_seconds': wall_time,
        'total_trials': store.count(seed=seed),
        'valid_trials': len(valid_df),
        'population_size': GA_POP_SIZE,
    }


def run_ga_experiment() -> dict:
    """Run GA benchmark for all seeds using the same device and objective."""
    if not RUN_GA_BENCHMARK:
        print('RUN_GA_BENCHMARK = False — skipping GA experiment')
        return None

    matched_trials = GA_TOTAL_TRIALS if GA_TOTAL_TRIALS is not None else BASELINE_TOTAL_TRIALS

    print('=' * 70)
    print('v3: GENETIC ALGORITHM BASELINE EXPERIMENT')
    print('=' * 70)
    print(f'  Seeds             : {SEEDS[:N_SEEDS]}')
    print(f'  Population size   : {GA_POP_SIZE}')
    print(f'  Trials per seed   : {matched_trials} (+ boundaries if enabled)')
    print(f'  Elitism           : {GA_ELITISM}')
    print(f'  Mutation/Cross    : {GA_MUTATION_RATE:.2f} / {GA_CROSSOVER_RATE:.2f}')
    print('=' * 70)

    store = TrialStore(GA_DB_PATH, device.feature_cols)
    runner = ScapsRunner(SCAPS_EXE, WORKING_DIR, SCRIPT_FILE,
                         OUTPUT_IV_FILE, OUTPUT_QE_FILE, timeout=MAX_TIMEOUT)

    per_seed = []
    for seed in SEEDS[:N_SEEDS]:
        r = run_ga_single_seed(seed, matched_trials, device, runner, store)
        per_seed.append(r)

    champions = [r['global_best'] for r in per_seed]
    ms = multi_seed_statistics(champions)
    total_wall = sum(r['wall_time_seconds'] for r in per_seed)

    print('\n' + '═'*70)
    print('GA SUMMARY')
    print('═'*70)
    print(f'  Champions per seed: {[f"{c:.3f}" for c in champions]}')
    print(f'  Mean ± std        : {ms["mean"]:.4f} ± {ms["std"]:.4f}%')
    print(f'  95% CI            : [{ms["ci_95_lo"]:.4f}, {ms["ci_95_hi"]:.4f}]%')
    print(f'  Total wall time   : {total_wall/3600:.2f} hours')

    summary = {
        'per_seed': per_seed,
        'multi_seed': ms,
        'store': store,
        'total_wall_seconds': total_wall,
        'matched_trials_per_seed': matched_trials,
        'ga_config': {
            'population_size': GA_POP_SIZE,
            'elitism': GA_ELITISM,
            'mutation_rate': GA_MUTATION_RATE,
            'crossover_rate': GA_CROSSOVER_RATE,
            'tournament_k': GA_TOURNAMENT_K,
            'mutation_sigma': GA_MUTATION_SIGMA,
        },
    }

    with open(os.path.join(OUT_DIR, 'ga_summary.json'), 'w') as f:
        json.dump({
            'multi_seed': ms,
            'per_seed': [{k: v for k, v in r.items() if k != 'best_per_generation'} for r in per_seed],
            'ga_config': summary['ga_config'],
            'total_wall_hours': total_wall / 3600,
            'matched_trials_per_seed': matched_trials,
        }, f, indent=2)

    return summary


print('GA runner defined. Do NOT call manually.')
print('If you want the GA benchmark, run the dedicated optional GA cell added below.')


# ####################################################################################
# From the original notebook
# ####################################################################################
# CELL 16 — Parameter sweeps (save .scaps per temperature + full IV curves)
# ============================================================
# Notes:
#   - Sweep points: 14/11/10
#   - Save .scaps file per temperature point (can open in SCAPS GUI)
#   - Save full IV curves for every Rs and Rsh sweep point
#   - Auto-extract coefficients: dVoc/dT, dPCE/dT, critical Rs, critical Rsh
#   - JSON output of extracted analysis for paper methods section
# ============================================================

import shutil
import re as _re

def _save_scaps_with_temperature(source_scaps, target_scaps, temp_k):
    """Copy .scaps file with modified temperature fields.
    
    The .scaps format stores Temperature and Secondworkpoint Temperature.
    We modify both to match the sweep point.
    """
    with open(source_scaps, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Replace temperature (both fields)
    content = _re.sub(
        r'Temperature :\s+[\d.]+\s*K',
        f'Temperature :  {temp_k:.2f} K',
        content
    )
    
    with open(target_scaps, 'w', encoding='utf-8') as f:
        f.write(content)


def _extract_temperature_coefficients(temp_data):
    """Extract dVoc/dT, dPCE/dT from linear fit."""
    if not temp_data or len(temp_data) < 3:
        return {}
    
    df = pd.DataFrame(temp_data)
    from scipy.stats import linregress
    
    r_voc = linregress(df['T_K'], df['Voc'])
    r_eta = linregress(df['T_K'], df['eta'])
    r_jsc = linregress(df['T_K'], df['Jsc'])
    r_ff  = linregress(df['T_K'], df['FF'])
    
    return {
        'dVoc_dT_mV_per_K': r_voc.slope * 1000,
        'dPCE_dT_percent_per_K': r_eta.slope,
        'dJsc_dT_mA_per_K': r_jsc.slope,
        'dFF_dT_percent_per_K': r_ff.slope,
        'R2_Voc': r_voc.rvalue ** 2,
        'R2_PCE': r_eta.rvalue ** 2,
        'R2_Jsc': r_jsc.rvalue ** 2,
        'R2_FF':  r_ff.rvalue ** 2,
        'Voc_at_300K': float(df[df['T_K'] == 300]['Voc'].values[0]) if 300 in df['T_K'].values else None,
        'PCE_at_300K': float(df[df['T_K'] == 300]['eta'].values[0]) if 300 in df['T_K'].values else None,
    }


def _extract_rs_critical(rs_data, ff_threshold=75.0):
    """Find critical Rs where FF drops below threshold."""
    if not rs_data or len(rs_data) < 3:
        return {}
    
    df = pd.DataFrame(rs_data).sort_values('Rs').reset_index(drop=True)
    
    critical = None
    for i in range(len(df) - 1):
        if df.iloc[i]['FF'] >= ff_threshold > df.iloc[i + 1]['FF']:
            r1, f1 = df.iloc[i][['Rs', 'FF']]
            r2, f2 = df.iloc[i + 1][['Rs', 'FF']]
            critical = r1 + (r2 - r1) * (f1 - ff_threshold) / (f1 - f2)
            break
    
    return {
        'critical_Rs_ohm_cm2': float(critical) if critical else 'beyond_tested_range',
        'FF_threshold_percent': ff_threshold,
        'FF_at_min_Rs': float(df.iloc[0]['FF']),
        'FF_at_max_Rs': float(df.iloc[-1]['FF']),
        'PCE_at_min_Rs': float(df.iloc[0]['eta']),
        'PCE_at_max_Rs': float(df.iloc[-1]['eta']),
    }


def _extract_rsh_critical(rsh_data, pce_retention=0.95):
    """Find critical Rsh where PCE falls below retention*PCE_max."""
    if not rsh_data or len(rsh_data) < 3:
        return {}
    
    df = pd.DataFrame(rsh_data).sort_values('Rsh').reset_index(drop=True)
    pce_max = df['eta'].max()
    pce_threshold = pce_retention * pce_max
    
    critical = None
    df_sorted = df.sort_values('Rsh', ascending=False).reset_index(drop=True)
    for i in range(len(df_sorted) - 1):
        if df_sorted.iloc[i]['eta'] >= pce_threshold > df_sorted.iloc[i + 1]['eta']:
            r1, p1 = df_sorted.iloc[i][['Rsh', 'eta']]
            r2, p2 = df_sorted.iloc[i + 1][['Rsh', 'eta']]
            critical = r1 + (r2 - r1) * (p1 - pce_threshold) / (p1 - p2)
            break
    
    return {
        'critical_Rsh_ohm_cm2': float(critical) if critical else 'beyond_tested_range',
        'PCE_retention_threshold': pce_retention,
        'PCE_max': float(pce_max),
        'PCE_at_min_Rsh': float(df.iloc[0]['eta']),
        'PCE_at_max_Rsh': float(df.iloc[-1]['eta']),
    }


def run_parameter_sweeps(best_scaps_file, output_subdir=None):
    """Run T/Rs/Rsh sweeps with expanded points and .scaps saving."""
    if output_subdir is None:
        output_subdir = OUT_DIR
    elif not os.path.isabs(output_subdir):
        output_subdir = os.path.join(OUT_DIR, output_subdir)
    os.makedirs(output_subdir, exist_ok=True)
    
    results = {'temperature': [], 'rs': [], 'rsh': []}
    runner = ScapsRunner(SCAPS_EXE, WORKING_DIR, SCRIPT_FILE,
                          OUTPUT_IV_FILE, OUTPUT_QE_FILE, timeout=MAX_TIMEOUT)

    # -------- Temperature sweep --------
    temp_dir = os.path.join(output_subdir, 'temperature_sweep')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'\nTemperature sweep — {len(TEMP_SWEEP_POINTS)} points (250-380 K)...')
    
    for temp_k in TEMP_SWEEP_POINTS:
        if SAVE_SCAPS_PER_TEMPERATURE:
            sweep_scaps = os.path.join(temp_dir, f'best_T{temp_k}K.scaps')
            _save_scaps_with_temperature(best_scaps_file, sweep_scaps, temp_k)
        else:
            sweep_scaps = best_scaps_file
        
        write_scaps_script(SCRIPT_FILE, sweep_scaps, OUTPUT_IV_FILE, OUTPUT_QE_FILE,
                           temperature_k=temp_k)
        if runner.run():
            r = parse_iv(OUTPUT_IV_FILE)
            if r:
                Voc, Jsc, FF, eta = r
                results['temperature'].append(
                    {'T_K': temp_k, 'Voc': Voc, 'Jsc': Jsc, 'FF': FF, 'eta': eta})
                try:
                    shutil.copy(OUTPUT_IV_FILE, os.path.join(temp_dir, f'iv_curve_T{temp_k}K.csv'))
                except Exception: pass
                print(f'  T={temp_k}K: PCE={eta:.3f}%  Voc={Voc:.4f}V  FF={FF:.2f}%')

    # -------- Rs sweep --------
    rs_dir = os.path.join(output_subdir, 'rs_sweep')
    os.makedirs(rs_dir, exist_ok=True)
    print(f'\nRs sweep — {len(RS_SWEEP_POINTS)} points (0.1-10 Ohm.cm2)...')
    
    for rs_val in RS_SWEEP_POINTS:
        write_scaps_script(SCRIPT_FILE, best_scaps_file, OUTPUT_IV_FILE, OUTPUT_QE_FILE,
                           rs_val_ohm_cm2=rs_val)
        if runner.run():
            r = parse_iv(OUTPUT_IV_FILE)
            if r:
                Voc, Jsc, FF, eta = r
                results['rs'].append({'Rs': rs_val, 'Voc': Voc, 'Jsc': Jsc, 'FF': FF, 'eta': eta})
                try:
                    shutil.copy(OUTPUT_IV_FILE, os.path.join(rs_dir, f'iv_curve_Rs{rs_val:.2f}.csv'))
                    shutil.copy(SCRIPT_FILE, os.path.join(rs_dir, f'script_Rs{rs_val:.2f}.txt'))
                except Exception: pass
                print(f'  Rs={rs_val:.2f} Ohm.cm2: PCE={eta:.3f}%  FF={FF:.2f}%')

    # Write README for Rs sweep
    readme_rs_lines = [
        'Rs Sweep Output Files',
        '=====================',
        'Note: Rs (series resistance) is NOT stored in SCAPS .scaps files.',
        'It is set at runtime via "set external.Rs" command in the script.',
        '',
        'To reproduce any sweep point in SCAPS GUI:',
        '  1. Open optimal_device.scaps',
        '  2. In Action panel, manually set Rs to the desired value (Ohm.cm2)',
        '  3. Run IV calculation',
        '  4. Should match the corresponding iv_curve_Rs*.csv file',
        '',
        f'Sweep points (Ohm.cm2): {RS_SWEEP_POINTS}',
    ]
    with open(os.path.join(rs_dir, 'README.txt'), 'w') as f:
        f.write('\n'.join(readme_rs_lines))

    # -------- Rsh sweep --------
    rsh_dir = os.path.join(output_subdir, 'rsh_sweep')
    os.makedirs(rsh_dir, exist_ok=True)
    print(f'\nRsh sweep — {len(RSH_SWEEP_POINTS)} points (50-10000 Ohm.cm2)...')
    
    for rsh_val in RSH_SWEEP_POINTS:
        write_scaps_script(SCRIPT_FILE, best_scaps_file, OUTPUT_IV_FILE, OUTPUT_QE_FILE,
                           rsh_val_ohm_cm2=rsh_val)
        if runner.run():
            r = parse_iv(OUTPUT_IV_FILE)
            if r:
                Voc, Jsc, FF, eta = r
                results['rsh'].append({'Rsh': rsh_val, 'Voc': Voc, 'Jsc': Jsc, 'FF': FF, 'eta': eta})
                try:
                    shutil.copy(OUTPUT_IV_FILE, os.path.join(rsh_dir, f'iv_curve_Rsh{rsh_val:.0f}.csv'))
                    shutil.copy(SCRIPT_FILE, os.path.join(rsh_dir, f'script_Rsh{rsh_val:.0f}.txt'))
                except Exception: pass
                print(f'  Rsh={rsh_val} Ohm.cm2: PCE={eta:.3f}%  Voc={Voc:.4f}V')

    readme_rsh_lines = [
        'Rsh Sweep Output Files',
        '======================',
        'Note: Rsh (shunt resistance) is NOT stored in SCAPS .scaps files.',
        'It is set at runtime via "set external.Rsh" command.',
        '',
        'To reproduce any sweep point in SCAPS GUI:',
        '  1. Open optimal_device.scaps',
        '  2. In Action panel, manually set Rsh to the desired value (Ohm.cm2)',
        '  3. Run IV calculation',
        '  4. Should match the corresponding iv_curve_Rsh*.csv file',
        '',
        f'Sweep points (Ohm.cm2): {RSH_SWEEP_POINTS}',
    ]
    with open(os.path.join(rsh_dir, 'README.txt'), 'w') as f:
        f.write('\n'.join(readme_rsh_lines))

    # -------- Extract analysis coefficients --------
    analysis = {
        'temperature': _extract_temperature_coefficients(results['temperature']),
        'rs': _extract_rs_critical(results['rs'], ff_threshold=75.0),
        'rsh': _extract_rsh_critical(results['rsh'], pce_retention=0.95),
    }
    results['analysis'] = analysis
    
    analysis_file = os.path.join(output_subdir, 'sweep_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump({
            'temperature_points': results['temperature'],
            'rs_points': results['rs'],
            'rsh_points': results['rsh'],
            'analysis': analysis,
            'metadata': {
                'champion_scaps': best_scaps_file,
                'n_temperature_points': len(results['temperature']),
                'n_rs_points': len(results['rs']),
                'n_rsh_points': len(results['rsh']),
            }
        }, f, indent=2)
    print(f'\n  Sweep analysis saved: {analysis_file}')
    
    if analysis['temperature']:
        print(f'\n  Temperature coefficients (paper-ready):')
        t = analysis['temperature']
        print(f'    dVoc/dT = {t.get("dVoc_dT_mV_per_K", 0):.3f} mV/K  (R2={t.get("R2_Voc", 0):.3f})')
        print(f'    dPCE/dT = {t.get("dPCE_dT_percent_per_K", 0):.4f} %/K  (R2={t.get("R2_PCE", 0):.3f})')

    return results


def plot_parameter_sweeps(results, out_path):
    """12-panel sweep plot with coefficient annotations."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle('Parameter Sensitivity — Optimal Device', 
                 fontweight='bold', fontsize=14)
    
    configs = [
        ('temperature', 'T_K',  'Temperature (K)'),
        ('rs',          'Rs',   'Rs (Ohm.cm2)'),
        ('rsh',         'Rsh',  'Rsh (Ohm.cm2)'),
    ]
    metrics = [('eta', 'PCE (%)'), ('Voc', 'Voc (V)'), 
               ('Jsc', 'Jsc (mA/cm2)'), ('FF', 'FF (%)')]
    colors = ['#1565C0', '#D84315', '#2E7D32']
    
    for i, (key, x_col, x_label) in enumerate(configs):
        if not results.get(key): continue
        df = pd.DataFrame(results[key])
        for j, (metric, y_label) in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(df[x_col], df[metric], 'o-', color=colors[i], lw=2, ms=7)
            ax.set_xlabel(x_label); ax.set_ylabel(y_label)
            if key in ('rs', 'rsh'):
                ax.set_xscale('log')
            ax.grid(alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)
    
    # Annotate temperature coefficient
    analysis = results.get('analysis', {})
    if analysis.get('temperature'):
        t = analysis['temperature']
        axes[0, 0].text(0.05, 0.95,
            f'dPCE/dT = {t.get("dPCE_dT_percent_per_K", 0):.3f} %/K',
            transform=axes[0, 0].transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved plot: {out_path}')


print('Parameter sweep functions ready.')
print(f'  Points: T={len(TEMP_SWEEP_POINTS)} / Rs={len(RS_SWEEP_POINTS)} / Rsh={len(RSH_SWEEP_POINTS)}')
print(f'  Save .scaps per temperature: {SAVE_SCAPS_PER_TEMPERATURE}')


# ####################################################################################
# From the original notebook
# ####################################################################################

# Monte Carlo fabrication tolerance (used by notebook cell 7.1)
# ============================================================
#   - a full .scaps file is written for each Monte Carlo sample
#   - nominal values are read from the trials database
#   - each variable is drawn from a log-normal distribution: value * exp(N(0, ln(1 + sigma)))
#   - the physics precheck is applied before SCAPS-1D
#   - raw samples (CSV) and a summary (JSON) are saved
# ============================================================

def _sample_parameter_v32(nominal_value, sigma, distribution):
    """Sample a perturbed parameter value."""
    if distribution == 'lognormal':
        log_sigma = np.log1p(sigma)
        return nominal_value * np.exp(np.random.normal(0, log_sigma))
    if distribution == 'normal':
        return nominal_value + np.random.normal(0, sigma)
    raise ValueError(f'Unknown distribution: {distribution}')


def _get_mc_nominal_params():
    """Recover nominal optimal parameters from trials DB, then JSON, then defaults."""
    etl = device.get_layer_by_role('ETL')
    absorber = device.get_layer_by_role('ABSORBER')
    htl = device.get_layer_by_role('HTL')
    defaults = {
        f'{etl.name}_thickness'      : 0.050 if etl else 0.050,
        f'{absorber.name}_thickness' : 0.350 if absorber else 0.350,
        f'{htl.name}_thickness'      : 0.100 if htl else 0.100,
        f'{etl.name}_ND'             : 1.0e18 if etl else 1.0e18,
        f'{htl.name}_NA'             : 1.0e18 if htl else 1.0e18,
        f'{absorber.name}_defect'    : 1.0e14 if absorber else 1.0e14,
        'back_contact_wf'            : VALIDATED_BACK_WF,
    }
    for iface_key in getattr(device, 'interface_space', {}):
        defaults[iface_key] = FIXED_INTERFACE_N

    db_path = globals().get('DB_PATH', os.path.join(OUT_DIR, 'trials.db'))
    try:
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql('SELECT * FROM trials WHERE valid_run = 1 ORDER BY eta DESC LIMIT 1', conn)
            if len(df):
                row = df.iloc[0].to_dict()
                for _layer in [etl, absorber, htl]:
                    if _layer is not None:
                        _short = f'{_layer.name}_t'
                        _legacy = f'{_layer.name}_thickness'
                        if _short in row and pd.notna(row[_short]):
                            row[_legacy] = row[_short]
                defaults.update({k: v for k, v in row.items() if k in defaults and pd.notna(v)})
                return defaults
    except Exception as e:
        print(f'  WARN: could not recover optimal device from trials DB: {e}')

    summary_file = os.path.join(OUT_DIR, 'experiment_summary.json')
    try:
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            for seed_row in summary.get('per_seed', []):
                pass
    except Exception:
        pass

    return defaults


def _mc_build_updates(perturbed):
    etl = device.get_layer_by_role('ETL')
    absorber = device.get_layer_by_role('ABSORBER')
    htl = device.get_layer_by_role('HTL')

    layer_updates = {}
    if etl:
        layer_updates[etl.block_name] = {
            'thickness_um': perturbed.get(f'{etl.name}_thickness'),
            'doping_type' : etl.doping_type,
            'na_val'      : None,
            'nd_val'      : perturbed.get(f'{etl.name}_ND'),
            'defect'      : None,
        }
    if absorber:
        layer_updates[absorber.block_name] = {
            'thickness_um': perturbed.get(f'{absorber.name}_thickness'),
            'doping_type' : absorber.doping_type,
            'na_val'      : FIXED_ABSORBER_NA,
            'nd_val'      : FIXED_ABSORBER_ND,
            'defect'      : perturbed.get(f'{absorber.name}_defect'),
        }
    if htl:
        layer_updates[htl.block_name] = {
            'thickness_um': perturbed.get(f'{htl.name}_thickness'),
            'doping_type' : htl.doping_type,
            'na_val'      : perturbed.get(f'{htl.name}_NA'),
            'nd_val'      : None,
            'defect'      : None,
        }

    iface_vals = {}
    for key in getattr(device, 'interface_space', {}):
        iface_vals[key] = perturbed.get(key, FIXED_INTERFACE_N)

    return layer_updates, iface_vals


def run_mc_uncertainty(optimal_device_file):
    """Run full structural Monte Carlo uncertainty with .scaps mutation."""
    print('=' * 70)
    print(f'MONTE CARLO UNCERTAINTY — {MC_N_SAMPLES} samples')
    print('=' * 70)

    if not os.path.exists(optimal_device_file):
        print(f'  ERROR: Optimal device file not found: {optimal_device_file}')
        return None

    nominal = _get_mc_nominal_params()

    # Map legacy sigma names to actual notebook column names
    etl = device.get_layer_by_role('ETL')
    absorber = device.get_layer_by_role('ABSORBER')
    htl = device.get_layer_by_role('HTL')
    iface_keys = list(getattr(device, 'interface_space', {}).keys())

    sigma_map = {
        'wo3_thickness'     : f'{etl.name}_thickness' if etl else 'WO3_thickness',
        'absorber_thickness': f'{absorber.name}_thickness' if absorber else 'MASnIBr2_thickness',
        'cbts_thickness'    : f'{htl.name}_thickness' if htl else 'CBTS_thickness',
        'wo3_ND'            : f'{etl.name}_ND' if etl else 'WO3_ND',
        'cbts_NA'           : f'{htl.name}_NA' if htl else 'CBTS_NA',
        'absorber_Nt'       : f'{absorber.name}_defect' if absorber else 'MASnIBr2_defect',
    }
    if len(iface_keys) >= 1:
        sigma_map['iface1_Nt'] = iface_keys[0]
    if len(iface_keys) >= 2:
        sigma_map['iface2_Nt'] = iface_keys[1]

    runner = ScapsRunner(SCAPS_EXE, WORKING_DIR, SCRIPT_FILE,
                         OUTPUT_IV_FILE, OUTPUT_QE_FILE, timeout=MAX_TIMEOUT)

    mc_dir = os.path.join(OUT_DIR, 'mc_uncertainty')
    os.makedirs(mc_dir, exist_ok=True)

    np.random.seed(42)
    mc_rows = []
    rejected_precheck = 0
    rejected_scaps = 0

    for i in range(MC_N_SAMPLES):
        perturbed = {}
        for legacy_key, (sigma, dist) in MC_INPUT_SIGMA.items():
            key = sigma_map.get(legacy_key, legacy_key)
            nominal_val = nominal.get(key)
            if nominal_val is None:
                continue
            perturbed[key] = _sample_parameter_v32(nominal_val, sigma, dist)

        # Keep back-contact fixed unless explicitly present
        perturbed['back_contact_wf'] = nominal.get('back_contact_wf', VALIDATED_BACK_WF)

        layer_updates, iface_vals = _mc_build_updates(perturbed)

        if MC_USE_PHYSICS_PRECHECK:
            check = physics_precheck_detailed(layer_updates, iface_vals, device)
            if not check['valid']:
                rejected_precheck += 1
                continue

        temp_scaps = os.path.join(mc_dir, f'mc_sample_{i:04d}.scaps')
        try:
            build_updated_scaps_file(
                TEMPLATE_FILE, temp_scaps, layer_updates, iface_vals,
                back_contact_wf=perturbed['back_contact_wf'], verify=False
            )
        except Exception:
            rejected_scaps += 1
            continue

        write_scaps_script(
            SCRIPT_FILE, temp_scaps, OUTPUT_IV_FILE, OUTPUT_QE_FILE,
            run_qe=False
        )
        ok = runner.run(wait_for_qe=False)
        if not ok:
            rejected_scaps += 1
            continue

        parsed = parse_iv(OUTPUT_IV_FILE)
        if parsed is None:
            rejected_scaps += 1
            continue

        Voc, Jsc, FF, eta = parsed
        row = {
            'sample_idx': i,
            'Voc': Voc,
            'Jsc': Jsc,
            'FF': FF,
            'eta': eta,
        }
        row.update(perturbed)
        mc_rows.append(row)

    if not mc_rows:
        print('  ERROR: no successful MC samples')
        return None

    mc_df = pd.DataFrame(mc_rows)
    stats_dict = {
        'n_success': int(len(mc_df)),
        'n_requested': int(MC_N_SAMPLES),
        'n_rejected_precheck': int(rejected_precheck),
        'n_rejected_scaps': int(rejected_scaps),
        'yield_percent': float(100 * len(mc_df) / max(MC_N_SAMPLES, 1)),
        'eta_mean': float(mc_df['eta'].mean()),
        'eta_std': float(mc_df['eta'].std(ddof=1) if len(mc_df) > 1 else 0.0),
        'eta_p05': float(mc_df['eta'].quantile(0.05)),
        'eta_p50': float(mc_df['eta'].quantile(0.50)),
        'eta_p95': float(mc_df['eta'].quantile(0.95)),
        'Voc_mean': float(mc_df['Voc'].mean()),
        'Jsc_mean': float(mc_df['Jsc'].mean()),
        'FF_mean': float(mc_df['FF'].mean()),
        'champion_scaps': optimal_device_file,
    }

    mc_df.to_csv(os.path.join(mc_dir, 'mc_samples.csv'), index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(mc_df['eta'], bins=30, alpha=0.85)
    plt.axvline(stats_dict['eta_mean'], linestyle='--', linewidth=2, label=f"mean = {stats_dict['eta_mean']:.2f}%")
    plt.axvline(stats_dict['eta_p05'], linestyle=':', linewidth=2, label=f"p05 = {stats_dict['eta_p05']:.2f}%")
    plt.axvline(stats_dict['eta_p95'], linestyle=':', linewidth=2, label=f"p95 = {stats_dict['eta_p95']:.2f}%")
    plt.xlabel('PCE (%)')
    plt.ylabel('Count')
    plt.title(f'Monte Carlo PCE distribution under fabrication uncertainty (N={MC_N_SAMPLES})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mc_dir, 'mc_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(mc_dir, 'mc_uncertainty_v32.json'), 'w', encoding='utf-8') as f:
        json.dump({'stats': stats_dict}, f, indent=2)

    print(f"  Success: {stats_dict['n_success']}/{stats_dict['n_requested']} samples")
    print(f"  PCE mean ± std: {stats_dict['eta_mean']:.3f} ± {stats_dict['eta_std']:.3f}%")
    print(f"  PCE p05/p95:   {stats_dict['eta_p05']:.3f}% / {stats_dict['eta_p95']:.3f}%")
    print(f"  Saved: {os.path.join(mc_dir, 'mc_samples.csv')}")
    print(f"  Saved: {os.path.join(mc_dir, 'mc_uncertainty_v32.json')}")

    return {'stats': stats_dict, 'samples': mc_df}


# ####################################################################################
# From the original notebook
# ####################################################################################
import os
import pandas as pd
import sqlite3
import json

print("="*70)
print("  RELOADING SAVED RESULTS FROM DISK")
print("="*70)

def reload_optimization_results(db_path, label):
    if not os.path.exists(db_path):
        print(f"  {label}: NOT FOUND at {db_path}")
        return None
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM trials WHERE valid_run=1', conn)
    conn.close()
    
    per_seed_list = []
    seeds_in_db = sorted(df['seed'].unique())
    for s in seeds_in_db:
        seed_df = df[df['seed'] == s]
        best = seed_df['eta'].max()
        per_seed_list.append({'seed': int(s), 'global_best': float(best)})
    
    # Build the champions string OUTSIDE the f-string (avoids the escape issue)
    champs_str = ', '.join([f"{p['global_best']:.4f}" for p in per_seed_list])
    
    print(f"  {label}: {len(df)} valid trials, {len(seeds_in_db)} seeds")
    print(f"    Champions: [{champs_str}]")
    
    if label == 'PPSR':
        return {'per_seed_results': per_seed_list, 'store': None}
    else:
        return {'per_seed': per_seed_list, 'store': None}
