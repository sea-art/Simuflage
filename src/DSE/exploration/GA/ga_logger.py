import logging
import statistics

LOG_LABELS = ['gen',
              'avg_ttf', 'avg_pe', 'avg_size',
              'max_ttf', 'max_pe', 'max_size',
              'stdev_ttf', 'stdev_pe', 'stdev_size']


class LoggerGA:
    def __init__(self, name, log_file, level=logging.DEBUG):
        """To setup as many loggers as you want"""

        open(log_file, 'w').close()

        handler = logging.FileHandler(log_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(handler)

        self.logger.info(", ".join(LOG_LABELS))

    def log(self, generation, gen_values):
        log_vals = [str(generation),
                    *list(map(str, self.log_mean_values(gen_values))),
                    *list(map(str, self.log_max_values(gen_values))),
                    *list(map(str, self.log_stdev_values(gen_values)))
                    ]

        assert len(LOG_LABELS) == len(log_vals), log_vals

        self.logger.info(", ".join(log_vals))


    @staticmethod
    def log_mean_values(gen_values):
        return tuple(statistics.mean(vals) for vals in gen_values)

    @staticmethod
    def log_max_values(gen_values):
        return tuple(max(vals) for vals in gen_values)

    @staticmethod
    def log_stdev_values(gen_values):
        return tuple(statistics.stdev(vals) for vals in gen_values)
