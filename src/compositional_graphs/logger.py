import yaml

class CustomLogger():
    def __init__(self):
        self.reset()

    def reset(self):
        self._results = []
        # self._episode = 0
    

    def log(self, setup_dict: dict, comp_time: dict, abs_dict: dict):
        """
         This method appends the computation results along with abstraction construciton results.
        """
        self._results.append({'setup': setup_dict, 'abs_dict': abs_dict, 'comp_time': comp_time})
    
    def dump_results_to_yaml(self, file_path: str, add_time_stamp: bool = True):
        """
        Dump the _results list to a YAML file.

        :param file_path: The path to the YAML file.
        """
        if add_time_stamp:
            import datetime
            now = datetime.datetime.now()
            timestamp: str = now.strftime("%Y%m%d_%H%M%S")
            file_path += f"_{timestamp}.yaml"
        else:
            file_path += ".yaml"
        tmp_dict = {f'Run {run}': run_data for run, run_data in enumerate(self._results)}
        with open(file_path, 'w') as file:
            yaml.dump(tmp_dict, file, default_flow_style=False)
