from typing import Union, List, Iterable
import re
import copy


class TeXTranslator:
    def __init__(self):
        self.bias_regex = re.compile("bg(ph|sp)([0-9])")
        self.probe_regex = re.compile(r"(WL|GCph|GCsp)")
        self.column_name_regex = re.compile(r" (FoM|Omm|Omb|h|ns|sigma8|w0|wa) ")
        self.probe_name_to_tex = {
            'Lensing': r'\mathrm{WL}', 'WL': r'\mathrm{WL}',
            'Void': r'\mathrm{V}', 'V': r'\mathrm{V}',
            'PhotometricGalaxy': r'\mathrm{GC_{ph}}', 'GCph': r'\mathrm{GC_{ph}}',
            'SpectroscopicGalaxy': r'\mathrm{GC_{sp}}', 'GCsp': r'\mathrm{GC_{sp}}',
            'GCsp(Pk)': r'\mathrm{GC_{sp}}(P_k)',
            'XC': r'\mathrm{XC}',
            'IST_GCsp': r'\mathrm{GC_{sp}}'
        }
        self.aliases_probes_name_to_tex = {
            "WL": r"\WL", "GCph": r"\GCph", "GCsp": r"\GCsp",
        }
        self.aliases_cosmo_par_name_to_tex = {
            "w0": r"\wz", "wa": r"\wa", "Omm": r"\Omm", "Omb": r"\Omb", "h": r"\h", "ns": r"\ns", "sigma8": r"\sige",
            "FoM": r"\FoM"
        }
        self.phys_par_name_to_tex = {
            'w0': r'w_0',
            'wa': r'w_a',
            'mnu': r'M_\nu[\mathrm{eV}]',
            'Omm': r'\Omega_m',
            'OmDE': r'\Omega_{DE}',
            'Omb': r'\Omega_b',
            'sigma8': r'\sigma_8',
            'ns': r'n_s',
            'h': r'h',
            'gamma': r'\gamma',
            'aIA': r'\mathcal{A}_{\rm IA}',
            'etaIA': r'\eta_{\rm IA}',
            'betaIA': r'\beta_{\rm IA}',
            'FoM': r'\mathrm{FoM}'
        }
        self.cosmology_name_to_tex = {
            'LCDM': r'\Lambda\mathrm{CDM}',
            'mnu_LCDM': r'\nu\Lambda\mathrm{CDM}', 'mnuLCDM': r'\nu\Lambda\mathrm{CDM}',
            'w0_wa_CDM': r'w_0w_a\mathrm{CDM}', 'w0waCDM': r'w_0w_a\mathrm{CDM}',
            'mnu_w0_wa_CDM': r'\nu w_0w_a\mathrm{CDM}', 'mnuw0waCDM': r'\nu w_0w_a\mathrm{CDM}'
        }
        self.probe_tex_to_name = {v: k for k, v in self.probe_name_to_tex.items()}
        self.cosmo_par_tex_to_name = {v: k for k, v in self.phys_par_name_to_tex.items()}
        self.cosmology_tex_to_name = {v: k for k, v in self.cosmology_name_to_tex.items()}

    def ProbeNameToTeX(self, name: "str") -> "str":
        if '+' not in name:
            tex_name = rf'{self.probe_name_to_tex[name]}'
        else:
            probes = [p.strip() for p in name.split('+')]
            probe_comb_tex_string = r'+'.join([r'%s' % self.probe_name_to_tex[p] for p in probes])
            tex_name = rf'{probe_comb_tex_string}'
        return tex_name

    def ProbeTeXToName(self, tex_name: str) -> str:
        if '+' not in tex_name:
            name = self.probe_name_to_tex[tex_name]
        else:
            probes = [self.probe_tex_to_name[p.strip()] for p in tex_name.split('+')]
            name = r'+'.join(probes)
        return name

    def CosmoParNameToTeX(self, name: str) -> str:
        return self.phys_par_name_to_tex[name]

    def CosmoParTeXToName(self, tex_name: str) -> str:
        return self.cosmo_par_tex_to_name[tex_name]

    def PhysParNameToTeX(self, s: "str", include_dollar: "bool" = False) -> "str":
        result = ""
        if "_" in s:
            result = " ".join([self.PhysParNameToTeX(p) for p in s.split("_")])
        try:
            result = self.CosmoParNameToTeX(s)
        except KeyError:
            match = self.bias_regex.match(s)
            if match:
                bias_type, idx = match.groups()
                result = r"b^{\mathrm{%s}}_{%s}" % (bias_type, idx)
            else:
                raise KeyError(f"Unrecognized physical parameter {s}")
        finally:
            if include_dollar:
                result = "$%s$" % result

            return result

    def PhysParsNamesToTeX(self, names: "Iterable[str]", include_dollar: "bool" = False) -> "Iterable[str]":
        tex_names = [self.PhysParNameToTeX(name, include_dollar=include_dollar) for name in names]
        if isinstance(names, set):
            tex_names = set(tex_names)
        return tex_names

    def CosmologyNameToTeX(self, cosmology_name: str) -> str:
        return self.cosmology_name_to_tex[cosmology_name]

    def CosmologyTeXToName(self, cosmology_tex_name: str) -> str:
        return self.cosmology_tex_to_name[cosmology_tex_name]

    def replaceFisherNamesToTeX(self, text: "str", use_aliases=False):
        result = text
        result = result.replace("Cl", r"C_{\ell}").replace("Pk", "P_k")

        if use_aliases:
            result = re.sub(r"XC\((WL|GCph|GCsp),(WL|GCph|GCsp)\)", r"\\xc{\1}{\2}", result)
            for p in set(self.probe_regex.findall(result)):
                result = result.replace(p, self.aliases_probes_name_to_tex[p])
        else:
            for p in set(self.probe_regex.findall(result)):
                result = result.replace(p, self.probe_name_to_tex[p])

            result = result.replace("XC", r"\mathrm{XC}")

        return result

    def replaceCosmoParsNamesToTeX(self, text: "str", use_aliases=False):
        result = text
        transl_dict = self.aliases_cosmo_par_name_to_tex if use_aliases else self.phys_par_name_to_tex

        result = self.column_name_regex.sub(lambda x: " %s " % transl_dict[x.groups()[0]], result)

        return result

    def translateToTeX(self, text: "str", use_aliases=False) -> "str":
        result = self.replaceFisherNamesToTeX(text, use_aliases=use_aliases)
        result = self.replaceCosmoParsNamesToTeX(result, use_aliases=use_aliases)

        return result

    def transformRawTeXTable(self, text: "str", use_aliases=True):
        result = re.sub(r"cline\{[0-9\-]+\}", "midrule", text)
        result = self.replaceFisherNamesToTeX(result, use_aliases=use_aliases)
        result = self.replaceCosmoParsNamesToTeX(result, use_aliases=use_aliases)
        result = result.replace(r"n\_bins", r"\GCsp\, \text{bins}")
        result = result.replace(r"+0.00\%", r"\text{--}")
        result = re.sub(r"(\+[0-9\.]+\\%)", r"\\positive{\1}", result)
        result = re.sub(r"(\-[0-9\.]+\\%)", r"\\negative{\1}", result)

        return result

    def toTeX(self, text: "str", use_aliases=False) -> "str":
        return self.translateToTeX(text=text, use_aliases=use_aliases)

    def translateLegendToTeX(self, legend):
        for label in legend.get_texts():
            tex_label = r"$%s$" % self.translateToTeX(label.get_text(), use_aliases=False)
            label.set_text(tex_label)
