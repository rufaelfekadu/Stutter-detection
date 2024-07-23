
class AverageMeter(object):
    def __init__(self, name=None, writer=None):
        self._writer = writer
        self._name = name
        self.reset()

    def reset(self):
        self.val = [0]
        self.avg = [0]
        self.sum = [0]
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        if isinstance(val, list):
            # Extend the lists if they are shorter than the incoming list
            length_difference = len(val) - len(self.sum)
            if length_difference > 0:
                self.sum.extend([0] * length_difference)
                self.avg.extend([0] * length_difference)
                self.val.extend([0] * length_difference)

            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = self.sum[i] / self.count
            self.val = val
        else:
            if not isinstance(self.sum, list):
                self.sum = [self.sum]
                self.avg = [self.avg]
                self.val = [self.val]
            self.sum[0] += val
            self.avg[0] = self.sum[0] / self.count
            self.val[0] = val

    def write(self, epoch=0):
        if self.count != 0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)

class LossMeter(object):
    def __init__(self, name=None, writer=None):
        self.reset()
        self._writer = writer
        self._name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, list):
            val = sum(val) / len(val)
            # log each value in the list
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def write(self, epoch=0):
        #  if self.avg is list then log each value
        if self.count !=0:
            if isinstance(self.avg, list):
                for i, val in enumerate(self.avg):
                    try:
                        self._writer.add_scalar(f'{self._name}_{i}', val, epoch)
                    except Exception as e:
                        print(e)
            else:
                try:
                    self._writer.add_scalar(self._name, self.avg, epoch)
                except Exception as e:
                    print(e)