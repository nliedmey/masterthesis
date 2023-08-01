import shutil


class StorageControl:

    def getStorageData(self):
        total, used, free = shutil.disk_usage('G:\\project')
        totalGB = total // (2 ** 30)
        usedGB = used // (2 ** 30)
        freeGB = free // (2 ** 30)
        return totalGB, usedGB, freeGB
