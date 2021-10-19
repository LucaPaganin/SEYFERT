from typing import List, Dict, Callable
from pathlib import Path
import re
import os
import logging

from seyfert.utils.type_helpers import TPathLike
from seyfert.utils.workspace import WorkSpace

logger = logging.getLogger(__name__)


class DirectoryExplorer:
    def __init__(self, root_path: "TPathLike"):
        self.root_path = Path(root_path)
        self.base_regexps = [
            re.compile(r"run_(?P<scenario>optimistic|pessimistic)"),
            re.compile(r"^(?P<scenario>optimistic|pessimistic)_(?P<nbins>[0-9]+)_sp_bins_"),
            re.compile(r"^scenario_(?P<scenario>optimistic|pessimistic)__n_sp_bins_(?P<nbins>[0-9]+)"),
        ]

    def getMandatoryDirectoryMatches(self, direc: "Path") -> "List[re.Match]":
        matches = []
        for regex in self.base_regexps:
            logger.info(f"Matching pattern {regex.pattern} to {direc.name}")
            match = regex.match(direc.name)
            if match:
                logger.info("Pattern matches, ok")
                matches.append(match)
                break
            else:
                logger.warning(f"{direc.name} does not match {regex.pattern}, trying next pattern")

        return matches

    def getDirMatches(self, direc: "Path", add_patterns: "List[str]" = None) -> "List[re.Match]":
        matches = self.getMandatoryDirectoryMatches(direc)
        if add_patterns is not None:
            matches += [re.search(pattern, direc.name) for pattern in add_patterns]

        return matches

    @staticmethod
    def directoryFilter(direc: "Path", matches: "List[re.Match]", add_filters: "List[Callable]" = None,
                        containing: "List[str]" = None, not_containing: "List[str]" = None) -> "bool":
        if matches:
            conds = [bool(m) for m in matches]
            if add_filters is not None:
                conds += [sel_func(direc.name) for sel_func in add_filters]
            if containing is not None:
                conds += [content in direc.name for content in containing]
            if not_containing is not None:
                conds += [content not in direc.name for content in not_containing]
            filter_passed = all(conds)
        else:
            filter_passed = False

        return filter_passed

    @staticmethod
    def getMatchesGroupsDictionary(matches: "List[re.Match]") -> "Dict":
        return {
            k: v for m in matches for k, v in m.groupdict().items()
        }

    def selectDirectories(self, recursive=False, add_patterns: "List[str]" = None, add_filters: "List[Callable]" = None,
                          containing: "List[str]" = None, not_containing: "List[str]" = None) -> "Dict[str, Dict]":
        dirs_map = {}
        if recursive:
            for root, dirs, files in os.walk(self.root_path):
                for direc in dirs:
                    direc = Path(root).resolve() / direc
                    matches = self.getDirMatches(direc, add_patterns=add_patterns)
                    filter_passed = self.directoryFilter(direc, matches, add_filters=add_filters,
                                                         containing=containing, not_containing=not_containing)
                    if filter_passed:
                        dirs_map[str(direc)] = self.getMatchesGroupsDictionary(matches)
        else:
            for direc in filter(lambda x: x.is_dir(), self.root_path.iterdir()):
                matches = self.getDirMatches(direc.resolve(), add_patterns=add_patterns)
                filter_passed = self.directoryFilter(direc, matches, add_filters=add_filters,
                                                     containing=containing, not_containing=not_containing)
                if filter_passed:
                    dirs_map[str(direc)] = self.getMatchesGroupsDictionary(matches)

        return dirs_map

    def selectWorkspaces(self, **kwargs) -> "List[WorkSpace]":
        direcs = self.selectDirectories(**kwargs)
        ws_list = []
        for direc in direcs:
            ws = WorkSpace(direc)
            ws.loadRunMetadata()
            ws_list.append(ws)

        return ws_list
