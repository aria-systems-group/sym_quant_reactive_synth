import yaml

# Representer for forcing lists to be in flow style (inline)
def flow_style_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, flow_style_list_representer)

class CustomLogger():
    def __init__(self):
        self.reset()

    def reset(self):
        self.status = None
        self.run_data = {}
        self.comp_time = {}
    

    def log(self, setup_dict: dict, comp_time: dict, abs_dict: dict):
        """
         This method appends the computation results along with abstraction construciton results.
        """
        self.run_data['Setup'] = setup_dict
        self.run_data['CompTime'] = comp_time
        self.run_data['AbsDict'] = abs_dict
        self.run_data['Status'] = self.status
    
    def dump_results_to_yaml(self, file_path: str, add_time_stamp: bool = True, iteration: int = None):
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
        # tmp_dict = {f'Run {run}': run_data for run, run_data in enumerate(self._results)}
        with open(file_path, 'a') as file:
            if iteration is not None:
                yaml.dump({f'Run {iteration}': self.run_data}, file, default_flow_style=False, sort_keys=False)
            else:
                yaml.dump(self.run_data, file, default_flow_style=False, sort_keys=False)
