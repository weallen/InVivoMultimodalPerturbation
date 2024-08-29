
from abc import ABCMeta, abstractmethodj

class AnalysisTask(ABCMeta):
    def run(self):
        raise NotImplementedError()
