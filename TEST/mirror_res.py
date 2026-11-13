import math
import re
import sys


# --- small vector helpers --------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    if n == 0.0:
        raise ValueError("zero-length vector; are N, CA and C collinear?")
    return (a[0] / n, a[1] / n, a[2] / n)


def _reflect(p, origin, normal):
    """Reflect point p through the plane with the given origin and normal."""
    d = _dot(_sub(p, origin), normal)
    return (p[0] - 2.0 * d * normal[0],
            p[1] - 2.0 * d * normal[1],
            p[2] - 2.0 * d * normal[2])


def _chirality_value(coords):
    """Signed triple product (N-CA) x (C-CA) . (CB-CA).

    Positive for a standard L-alpha-amino acid, negative for D.
    """
    ca = coords["CA"]
    return _dot(_cross(_sub(coords["N"], ca), _sub(coords["C"], ca)),
                _sub(coords["CB"], ca))


def _label(value):
    return "L" if value > 0 else "D"


# --- PDB column access -----------------------------------------------------

_COORD_RECORDS = ("ATOM  ", "HETATM")


def _is_coord(line):
    return line[:6] in _COORD_RECORDS


def _atom_name(line):
    return line[12:16].strip().upper()


def _alt_loc(line):
    return line[16]


def _res_name(line):
    return line[17:20].strip().upper()


def _chain_id(line):
    return line[21]


def _res_seq(line):
    return line[22:26].strip()


def _i_code(line):
    return line[26]


def _get_xyz(line):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _set_xyz(line, xyz):
    out = ""
    for v in xyz:
        if v == 0.0:
            v = 0.0
        t = f"{v:8.3f}"
        if len(t) > 8:
            raise ValueError(f"coordinate {v} does not fit the PDB field width")
        out += t
    return line[:30] + out + line[54:]


def _set_res_name(line, new_name):
    return line[:17] + f"{new_name:>3}" + line[20:]


# --- the class -------------------------------------------------------------

class ChiralityFlipper:
    """Invert the chirality of selected residues of a PDB file."""

    L_TO_D = {
        "ALA": "DAL", "ARG": "DAR", "ASN": "DSG", "ASP": "DAS", "CYS": "DCY",
        "GLN": "DGN", "GLU": "DGL", "HIS": "DHI", "ILE": "DIL", "LEU": "DLE",
        "LYS": "DLY", "MET": "MED", "PHE": "DPN", "PRO": "DPR", "SER": "DSN",
        "THR": "DTH", "TRP": "DTR", "TYR": "DTY", "VAL": "DVA",
    }
    D_TO_L = {d: l for l, d in L_TO_D.items()}

    ALIASES = {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS",
        "HSE": "HIS", "HSP": "HIS", "CYX": "CYS", "CYM": "CYS",
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    }

    # Atoms that define the mirror plane or belong to the backbone: never moved.
    FIXED_ATOMS = {
        "N", "CA", "C", "O", "OXT", "OT1", "OT2",
        # hydrogens on the backbone nitrogen
        "H", "HN", "H1", "H2", "H3", "HT1", "HT2", "HT3", "H01", "D", "DN",
    }

    # "A:5", "A5", "5", "5A", "A:20-24", "20-24"
    _ONE = re.compile(r"^\s*(?:([A-Za-z0-9])\s*:\s*)?"
                      r"(?:([A-Za-z])\s*)?(-?\d+)\s*([A-Za-z])?\s*$")
    _RANGE = re.compile(r"^\s*(?:([A-Za-z0-9])\s*:\s*)?"
                        r"(-?\d+)\s*-\s*(-?\d+)\s*$")

    def __init__(self,
                 input_pdb,
                 output_pdb="output_flipped.pdb",
                 targets=(),
                 chain="A",
                 rename=False,
                 verbose=True):
        # --- configuration, all of it given here -------------------------
        self.input_pdb = str(input_pdb)
        self.output_pdb = str(output_pdb)
        self.targets = targets
        self.default_chain = chain
        self.rename = bool(rename)
        self.verbose = bool(verbose)

        # --- state filled in while running -------------------------------
        self.lines = []        # the whole file, edited in place
        self.requests = []     # parsed targets: (chain, resseq, icode)
        self.results = []      # one record per request
        self.errors = []       # human-readable problems
        self.written = False

        # --- scratch for the residue currently being processed ------------
        self._chain = ""
        self._resseq = ""
        self._icode = ""
        self._indices = []
        self._coords = {}
        self._name = ""
        self._canonical = ""
        self._ca = None
        self._normal = None
        self._moved = []
        self._before = 0.0
        self._after = 0.0
        self._new_name = ""
        self._direction = ""
        self._note = ""

    # -- target parsing ----------------------------------------------------

    def _parse_targets(self):
        """Turn self.targets into self.requests, a list of (chain, seq, icode)."""
        raw = self.targets
        if raw is None:
            raw = []
        if isinstance(raw, (str, int)):
            raw = [raw]

        seen = set()
        self.requests = []

        for item in raw:
            self._note = item
            parsed = []

            if isinstance(item, int):
                parsed = [(self.default_chain, str(item), "")]

            elif isinstance(item, dict):
                parsed = [(str(item.get("chain", self.default_chain)),
                           str(item["resseq"]),
                           str(item.get("icode", "")))]

            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    parsed = [(str(item[0]), str(item[1]), "")]
                elif len(item) == 3:
                    parsed = [(str(item[0]), str(item[1]), str(item[2]))]
                else:
                    self.errors.append(f"cannot read target {item!r}")

            elif isinstance(item, str):
                m = self._RANGE.match(item)
                if m:
                    ch = m.group(1) or self.default_chain
                    lo, hi = int(m.group(2)), int(m.group(3))
                    if lo > hi:
                        lo, hi = hi, lo
                    parsed = [(ch, str(n), "") for n in range(lo, hi + 1)]
                else:
                    m = self._ONE.match(item)
                    if m:
                        ch = m.group(1) or m.group(2) or self.default_chain
                        parsed = [(ch, m.group(3), m.group(4) or "")]
                    else:
                        self.errors.append(f"cannot read target {item!r}")
            else:
                self.errors.append(f"cannot read target {item!r}")

            for key in parsed:
                norm = (key[0].strip(), key[1].strip(), key[2].strip())
                if norm in seen:
                    continue
                seen.add(norm)
                self.requests.append(norm)

        self._note = ""
        return bool(self.requests)

    # -- file I/O ----------------------------------------------------------

    def _load(self):
        try:
            with open(self.input_pdb) as fh:
                self.lines = fh.readlines()
        except OSError as exc:
            self.errors.append(f"could not read input_pdb: {exc}")
            return False
        return True

    def _write(self):
        done = [r for r in self.results if r["flipped"]]
        try:
            with open(self.output_pdb, "w") as fh:
                for r in done:
                    fh.write(f"REMARK   1 RESIDUE {r['name']} "
                             f"{r['chain']}/{r['resseq']}{r['icode']} "
                             f"CHIRALITY INVERTED ({r['direction']})\n")
                fh.write("REMARK   1 SIDE CHAINS REFLECTED THROUGH THEIR OWN "
                         "N-CA-C PLANES BY flip_residues.py\n")
                fh.writelines(self.lines)
        except OSError as exc:
            self.errors.append(f"could not write output_pdb: {exc}")
            return False
        self.written = True
        return True

    # -- one residue -------------------------------------------------------

    def _matches_current(self):
        """A closure-free filter over self.lines using the scratch target."""
        hits = []
        for i, ln in enumerate(self.lines):
            if not _is_coord(ln):
                continue
            if _chain_id(ln).strip() != self._chain.strip():
                continue
            if _res_seq(ln) != self._resseq:
                continue
            if _i_code(ln).strip() != self._icode.strip():
                continue
            hits.append(i)
        return hits

    def _collect_atoms(self):
        self._indices = self._matches_current()
        if not self._indices:
            self._note = "residue not found in the input file"
            return False

        self._name = _res_name(self.lines[self._indices[0]])
        self._canonical = self.ALIASES.get(self._name, self._name)

        altlocs = {_alt_loc(self.lines[i]) for i in self._indices}
        altlocs = {a for a in altlocs if a.strip()}
        if len(altlocs) > 1:
            self.errors.append(
                f"{self._name} {self._chain}/{self._resseq}: alternate "
                f"locations {sorted(altlocs)} all reflected through the plane "
                f"of the first one")

        self._coords = {}
        for i in self._indices:
            self._coords.setdefault(_atom_name(self.lines[i]),
                                    _get_xyz(self.lines[i]))
        return True

    def _validate(self):
        if self._canonical == "GLY":
            self._note = "glycine is achiral -- skipped"
            return False
        missing = [a for a in ("N", "CA", "C", "CB") if a not in self._coords]
        if missing:
            self._note = (f"missing {', '.join(missing)} -- cannot define "
                          f"the stereocentre")
            return False
        return True

    def _build_plane(self):
        self._ca = self._coords["CA"]
        try:
            self._normal = _unit(_cross(_sub(self._coords["N"], self._ca),
                                        _sub(self._coords["C"], self._ca)))
        except ValueError as exc:
            self._note = str(exc)
            return False
        self._before = _chirality_value(self._coords)
        return True

    def _apply(self):
        self._moved = []
        after_coords = dict(self._coords)
        for i in self._indices:
            a = _atom_name(self.lines[i])
            if a in self.FIXED_ATOMS:
                continue
            new = _reflect(_get_xyz(self.lines[i]), self._ca, self._normal)
            self.lines[i] = _set_xyz(self.lines[i], new)
            self._moved.append(a)
            after_coords[a] = new

        self._after = _chirality_value(after_coords)
        return bool(self._moved)

    def _resolve_name(self):
        """Decide the new residue name from the MEASURED chirality, not the
        current name -- a mislabelled file would otherwise be renamed wrongly.
        """
        self._new_name = self._name
        if self._before > 0:
            # the residue is L now, so it becomes D
            self._direction = "L -> D"
            if self._canonical in self.L_TO_D:
                self._new_name = self.L_TO_D[self._canonical]
            elif self._name in self.D_TO_L:
                # named as D but built as L: the name was already the right
                # one for the result, keep it
                self._new_name = self._name
        else:
            # the residue is D now, so it becomes L
            self._direction = "D -> L"
            if self._name in self.D_TO_L:
                self._new_name = self.D_TO_L[self._name]
            elif self._canonical in self.L_TO_D:
                self._new_name = self._name
            if self._name != self._new_name or self._canonical in self.L_TO_D:
                if self._name in self.L_TO_D or self._canonical in self.L_TO_D:
                    # named as L but built as D -- flag it, the name is only
                    # correct after the flip
                    self.errors.append(
                        f"{self._name} {self._chain}/{self._resseq}: named as "
                        f"L but built as D; flipping it makes it L and the "
                        f"existing name correct")

    def _rename_residue(self):
        if not self.rename or self._new_name == self._name:
            return
        for i in self._indices:
            self.lines[i] = _set_res_name(self.lines[i], self._new_name)

    def _record(self, flipped):
        renamed = (self.rename and flipped
                   and self._new_name != self._name)
        self.results.append({
            "chain": self._chain,
            "resseq": self._resseq,
            "icode": self._icode,
            "name": self._name,
            "flipped": flipped,
            "new_name": self._new_name if renamed else self._name,
            "renamed": renamed,
            "direction": self._direction,
            "moved": list(self._moved),
            "fixed": [a for a in self._coords if a in self.FIXED_ATOMS],
            "before": self._before,
            "after": self._after,
            "sign_flipped": flipped and ((self._before > 0) != (self._after > 0)),
            "note": self._note,
        })

    def _reset_scratch(self):
        self._indices = []
        self._coords = {}
        self._name = "?"
        self._canonical = "?"
        self._ca = None
        self._normal = None
        self._moved = []
        self._before = 0.0
        self._after = 0.0
        self._new_name = "?"
        self._direction = ""
        self._note = ""

    def _flip_current(self):
        """Flip the residue named by the scratch target. Returns True on success."""
        if not self._collect_atoms():
            self._record(False)
            return False
        if not self._validate():
            self._record(False)
            return False
        if not self._build_plane():
            self._record(False)
            return False
        self._resolve_name()
        if not self._apply():
            self._note = "no movable side-chain atoms found"
            self._record(False)
            return False
        self._rename_residue()
        self._record(True)
        return True

    def _flip_all(self):
        for chain, resseq, icode in self.requests:
            self._reset_scratch()
            self._chain, self._resseq, self._icode = chain, resseq, icode
            ok = self._flip_current()
            if not ok and self._note:
                self.errors.append(
                    f"{chain}/{resseq}{icode}: {self._note}")
        self._reset_scratch()

    # -- reporting ---------------------------------------------------------

    def _report(self):
        if not self.verbose:
            return
        done = [r for r in self.results if r["flipped"]]
        skipped = [r for r in self.results if not r["flipped"]]

        if self.written:
            print(f"wrote {self.output_pdb}")
        print(f"  requested : {len(self.results)}   "
              f"flipped : {len(done)}   skipped : {len(skipped)}")
        print()

        for r in done:
            tag = f"{r['name']} {r['chain']}/{r['resseq']}{r['icode']}"
            print(f"  {tag:<16} {r['direction']:<18} "
                  f"{_label(r['before'])} {r['before']:+8.3f}  ->  "
                  f"{_label(r['after'])} {r['after']:+8.3f}"
                  + ("" if r["sign_flipped"]
                     else "   WARNING: sign did not flip"))
            if r["renamed"]:
                print(f"    renamed to     : {r['new_name']}")
            print(f"    atoms moved    : {len(r['moved'])} "
                  f"({', '.join(r['moved'])})")
            print(f"    atoms held     : {', '.join(r['fixed'])}")

        if skipped:
            print()
            print("  skipped:")
            for r in skipped:
                tag = f"{r['name']} {r['chain']}/{r['resseq']}{r['icode']}"
                print(f"    {tag:<16} {r['note']}")

        if done:
            print()
            print("  Check the new side-chain positions for clashes with")
            print("  neighbours; the chi angles are mirrored and may need")
            print("  repacking or a short minimisation.")

    # -- entry point -------------------------------------------------------

    def run(self):
        """Parse the targets, flip them, write the file, print a report."""
        if not self._parse_targets():
            self.errors.append("no usable targets given")
        if self.errors and not self.requests:
            self._complain()
            return self.results
        if not self._load():
            self._complain()
            return self.results

        self._flip_all()

        if any(r["flipped"] for r in self.results):
            self._write()
        else:
            self.errors.append("nothing was flipped, so no file was written")

        self._report()
        self._complain()
        return self.results

    def _complain(self):
        if not self.errors or not self.verbose:
            return
        print()
        for e in self.errors:
            print(f"  note: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# SETTINGS -- edit these, then run:  python3 flip_residues.py
# ---------------------------------------------------------------------------

INPUT_PDB = "/Users/nikopol/Downloads/PROTEIN_JAR/new_cup.pdb"
OUTPUT_PDB = "output_flipped_d.pdb"

# Any mix of: 5  |  "5"  |  "A:5"  |  "A5"  |  "5A" (insertion code)
#             "20-24"  |  "B:20-24"  |  ("A", 5)  |  ("A", 5, "A")
TARGETS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

CHAIN = "A"      # default chain for targets that do not name one
RENAME = False   # rewrite residue names to the D codes (or back to L)


if __name__ == "__main__":
    flipper = ChiralityFlipper(
        input_pdb=INPUT_PDB,
        output_pdb=OUTPUT_PDB,
        targets=TARGETS,
        chain=CHAIN,
        rename=RENAME,
    )
    flipper.run()
    raise SystemExit(0 if any(r["flipped"] for r in flipper.results) else 1)