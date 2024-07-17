import matplotlib.pyplot as plt


def plot_accuracy(accuracy_by_iterations, used_section=None, path_to_save=None):
    plt.figure(figsize=(18, 10))
    plots_by_sections = {}
    iterations = []
    for _iter in accuracy_by_iterations.keys():
        iterations.append(_iter)
        for section in accuracy_by_iterations[_iter]:
            if used_section is None:
                if plots_by_sections.get(section['section'], None) is None:
                    plots_by_sections[section['section']] = []
                correct, incorrect = section['correct'], section['incorrect']
                plots_by_sections[section['section']].append(100.0 * correct / (correct + incorrect))
            else:
                if section['section'] in used_section:
                    if plots_by_sections.get(section['section'], None) is None:
                        plots_by_sections[section['section']] = []
                    correct, incorrect = section['correct'], section['incorrect']
                    plots_by_sections[section['section']].append(100.0 * correct / (correct + incorrect))

    for label, data in plots_by_sections.items():
        plt.plot(iterations, data, label=label)
    plt.xticks(iterations)
    plt.ylabel('accuracy, %')
    plt.xlabel('iteration')
    plt.legend(loc=0)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()
