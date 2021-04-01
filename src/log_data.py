import csv


class LogData:
    def __init__(self, folder="monitoring", filename_grads="grads.csv"):
        """
        filename = "data.csv", , filename_buffer = "data_buffer.csv", filename_compet = "comp_scores_data.csv"

        self.fieldnames = ['iter', 'loss', 'value_loss', 'prob_loss']
        self.fieldnames_buffer = ['iter', 'wins', 'losses', 'draws']
        self.fieldnames_compet = ['iter', 'scores']
        """
        self.folder = folder
        self.all_charts = {}

        self.filename_grads = filename_grads
        self.fieldnames_grads = ""
        self.write_headers_grads()

    def add_chart(self, chart_name, data):

        self.all_charts[chart_name] = data
        self.write_headers(chart_name)

    def write_headers(self, chart_name):

        filename = self.folder + "/" + self.all_charts[chart_name][0]
        fieldnames = self.all_charts[chart_name][1]

        with open(filename, "w") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    def save_data(self, chart_name, iter_number, data):

        filename = self.folder + "/" + self.all_charts[chart_name][0]
        fieldnames = self.all_charts[chart_name][1]
        stored_values = [iter_number] + list(data)

        with open(filename, "a") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            info = {k: v for (k, v) in zip(fieldnames, stored_values)}
            csv_writer.writerow(info)

    def write_headers_grads(self):

        layers = [
            "conv.weight",
            "fc1.weight",
            "fc2.weight",
            "fc_action1.weight",
            "fc_action2.weight",
            "fc_value1.weight",
            "fc_value2.weight",
        ]
        layers_average = [a + "_ave" for a in layers]
        layers_max = [a + "_max" for a in layers]
        layers_min = [a + "_min" for a in layers]

        fieldnames = ["iter"] + layers_average + layers_max + layers_min
        self.fieldnames_grads = fieldnames

        filename = self.folder + "/" + self.filename_grads

        with open(filename, "w") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

    def save_grads(self, iter_number, named_parameters):

        ave_grads = []
        max_grads = []
        min_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                ave_grads.append(float(p.grad.abs().mean()))
                max_grads.append(float(p.grad.abs().max()))
                min_grads.append(float(p.grad.abs().min()))

        stored_values = [iter_number] + ave_grads + max_grads + min_grads

        filename = self.folder + "/" + self.filename_grads

        with open(filename, "a") as csv_file:

            csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames_grads)
            info = {k: v for (k, v) in zip(self.fieldnames_grads, stored_values)}
            csv_writer.writerow(info)
