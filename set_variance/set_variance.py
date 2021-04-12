import json
import numpy as np
import random
import matplotlib.pyplot as plt

ref_data = open('../tddft_data/tddft_data.json')
ref_data = json.load(ref_data)

validation_set = ['step_1601_chromophore_21', 'step_1451_chromophore_25', 'step_401_chromophore_19', 'step_701_chromophore_2',
 'step_1001_chromophore_7', 'step_1601_chromophore_15', 'step_1251_chromophore_21', 'step_1701_chromophore_16',
 'step_1001_chromophore_5', 'step_1551_chromophore_10', 'step_351_chromophore_1', 'step_1151_chromophore_15',
 'step_701_chromophore_11', 'step_1651_chromophore_26', 'step_851_chromophore_20', 'step_151_chromophore_9',
 'step_1451_chromophore_6', 'step_1101_chromophore_3', 'step_351_chromophore_23', 'step_1501_chromophore_10',
 'step_401_chromophore_3', 'step_1_chromophore_12', 'step_1751_chromophore_23', 'step_351_chromophore_3',
 'step_1151_chromophore_13', 'step_1351_chromophore_18', 'step_1601_chromophore_17', 'step_1701_chromophore_14',
 'step_1_chromophore_13', 'step_101_chromophore_18', 'step_951_chromophore_5', 'step_251_chromophore_19',
 'step_1251_chromophore_6', 'step_1651_chromophore_2', 'step_551_chromophore_12', 'step_701_chromophore_26',
 'step_651_chromophore_27', 'step_551_chromophore_10', 'step_1701_chromophore_1', 'step_401_chromophore_14',
 'step_851_chromophore_3', 'step_501_chromophore_22', 'step_451_chromophore_22', 'step_251_chromophore_15',
 'step_501_chromophore_8', 'step_1301_chromophore_5', 'step_851_chromophore_14', 'step_651_chromophore_11',
 'step_1401_chromophore_8', 'step_551_chromophore_24', 'step_1701_chromophore_12', 'step_101_chromophore_19',
 'step_251_chromophore_22', 'step_151_chromophore_8', 'step_1001_chromophore_27', 'step_551_chromophore_14',
 'step_1_chromophore_11', 'step_401_chromophore_2', 'step_901_chromophore_21', 'step_301_chromophore_13',
 'step_1101_chromophore_16', 'step_1601_chromophore_11', 'step_901_chromophore_9', 'step_1051_chromophore_13',
 'step_1801_chromophore_17', 'step_1_chromophore_17', 'step_1051_chromophore_10', 'step_1151_chromophore_8',
 'step_151_chromophore_3', 'step_451_chromophore_5', 'step_1001_chromophore_22', 'step_1851_chromophore_26',
 'step_1451_chromophore_23', 'step_1151_chromophore_17', 'step_401_chromophore_15', 'step_1451_chromophore_21',
 'step_1_chromophore_23', 'step_1551_chromophore_15', 'step_801_chromophore_13', 'step_1701_chromophore_6',
 'step_951_chromophore_7', 'step_901_chromophore_14', 'step_1801_chromophore_22', 'step_1351_chromophore_16',
 'step_1801_chromophore_21', 'step_401_chromophore_12', 'step_951_chromophore_11', 'step_1451_chromophore_26',
 'step_801_chromophore_2', 'step_1701_chromophore_10', 'step_1201_chromophore_12', 'step_1301_chromophore_13',
 'step_651_chromophore_21', 'step_101_chromophore_23', 'step_1151_chromophore_24', 'step_801_chromophore_1',
 'step_51_chromophore_3', 'step_1101_chromophore_11', 'step_1751_chromophore_18', 'step_751_chromophore_20',
 'step_1201_chromophore_27', 'step_101_chromophore_7', 'step_701_chromophore_23', 'step_1551_chromophore_1',
 'step_1251_chromophore_9', 'step_1751_chromophore_8', 'step_501_chromophore_18', 'step_51_chromophore_17',
 'step_901_chromophore_19', 'step_1151_chromophore_23', 'step_1801_chromophore_11', 'step_251_chromophore_27',
 'step_1451_chromophore_3', 'step_1001_chromophore_11', 'step_1701_chromophore_19', 'step_901_chromophore_22',
 'step_1001_chromophore_8', 'step_1501_chromophore_26', 'step_1751_chromophore_24', 'step_801_chromophore_19',
 'step_451_chromophore_16', 'step_901_chromophore_13', 'step_651_chromophore_18', 'step_1851_chromophore_13',
 'step_251_chromophore_12', 'step_851_chromophore_27', 'step_951_chromophore_25', 'step_1651_chromophore_4',
 'step_801_chromophore_26', 'step_1551_chromophore_11', 'step_1101_chromophore_20', 'step_51_chromophore_18',
 'step_751_chromophore_8', 'step_201_chromophore_23', 'step_1301_chromophore_2', 'step_1201_chromophore_11',
 'step_451_chromophore_10', 'step_201_chromophore_7', 'step_501_chromophore_27', 'step_1501_chromophore_17',
 'step_51_chromophore_14', 'step_1051_chromophore_26', 'step_1801_chromophore_16', 'step_701_chromophore_7',
 'step_451_chromophore_26', 'step_1751_chromophore_2', 'step_1401_chromophore_11', 'step_251_chromophore_2',
 'step_751_chromophore_22', 'step_1351_chromophore_22', 'step_1201_chromophore_8', 'step_651_chromophore_23',
 'step_1101_chromophore_15', 'step_901_chromophore_8', 'step_351_chromophore_4', 'step_501_chromophore_11',
 'step_1451_chromophore_17', 'step_1001_chromophore_2', 'step_1601_chromophore_1', 'step_1151_chromophore_21',
 'step_201_chromophore_22', 'step_851_chromophore_22', 'step_1851_chromophore_19', 'step_901_chromophore_15',
 'step_1651_chromophore_12', 'step_201_chromophore_5', 'step_801_chromophore_20', 'step_1351_chromophore_26',
 'step_101_chromophore_24', 'step_1301_chromophore_24', 'step_1851_chromophore_12', 'step_1351_chromophore_6',
 'step_151_chromophore_23', 'step_501_chromophore_19', 'step_1051_chromophore_16', 'step_1_chromophore_24',
 'step_601_chromophore_6', 'step_1151_chromophore_5', 'step_301_chromophore_16', 'step_301_chromophore_22',
 'step_401_chromophore_10', 'step_851_chromophore_21', 'step_1351_chromophore_3', 'step_1201_chromophore_10',
 'step_1351_chromophore_12', 'step_1_chromophore_16', 'step_401_chromophore_11', 'step_1651_chromophore_6',
 'step_1101_chromophore_17', 'step_1551_chromophore_19', 'step_1551_chromophore_3', 'step_451_chromophore_9',
 'step_101_chromophore_10', 'step_851_chromophore_11', 'step_1751_chromophore_25', 'step_751_chromophore_4',
 'step_201_chromophore_10', 'step_601_chromophore_7', 'step_851_chromophore_10', 'step_1351_chromophore_1',
 'step_151_chromophore_26', 'step_1151_chromophore_12', 'step_1801_chromophore_12', 'step_951_chromophore_9',
 'step_451_chromophore_21', 'step_801_chromophore_21', 'step_1351_chromophore_4', 'step_851_chromophore_13',
 'step_1101_chromophore_19', 'step_351_chromophore_2', 'step_1201_chromophore_5', 'step_1151_chromophore_11',
 'step_1251_chromophore_2', 'step_1201_chromophore_25', 'step_851_chromophore_6', 'step_201_chromophore_24',
 'step_1051_chromophore_11', 'step_1401_chromophore_14', 'step_451_chromophore_2', 'step_301_chromophore_5',
 'step_1051_chromophore_23', 'step_1301_chromophore_25', 'step_201_chromophore_9', 'step_1651_chromophore_19',
 'step_1751_chromophore_16', 'step_1301_chromophore_21', 'step_1401_chromophore_20', 'step_401_chromophore_9',
 'step_651_chromophore_14', 'step_1851_chromophore_2', 'step_601_chromophore_16', 'step_251_chromophore_7',
 'step_51_chromophore_27', 'step_201_chromophore_14', 'step_1501_chromophore_15', 'step_251_chromophore_10',
 'step_1751_chromophore_20', 'step_1251_chromophore_1', 'step_1251_chromophore_7', 'step_1101_chromophore_24',
 'step_1801_chromophore_7', 'step_1401_chromophore_7', 'step_601_chromophore_5', 'step_1701_chromophore_17',
 'step_1251_chromophore_11', 'step_1551_chromophore_25', 'step_1251_chromophore_12', 'step_351_chromophore_6',
 'step_301_chromophore_4', 'step_551_chromophore_20', 'step_51_chromophore_1', 'step_1551_chromophore_24',
 'step_851_chromophore_25', 'step_551_chromophore_26', 'step_801_chromophore_3', 'step_1451_chromophore_19',
 'step_451_chromophore_19', 'step_1601_chromophore_4', 'step_151_chromophore_7', 'step_1_chromophore_21',
 'step_1651_chromophore_23', 'step_1251_chromophore_24', 'step_1101_chromophore_14', 'step_1151_chromophore_27',
 'step_1251_chromophore_14', 'step_601_chromophore_26', 'step_1851_chromophore_6', 'step_1501_chromophore_16',
 'step_301_chromophore_18', 'step_1151_chromophore_6', 'step_451_chromophore_18', 'step_401_chromophore_17',
 'step_1151_chromophore_16', 'step_351_chromophore_19', 'step_1501_chromophore_7', 'step_1351_chromophore_24',
 'step_1001_chromophore_10', 'step_251_chromophore_16', 'step_1651_chromophore_5', 'step_451_chromophore_1',
 'step_1651_chromophore_22', 'step_1701_chromophore_22', 'step_1301_chromophore_10', 'step_1751_chromophore_13',
 'step_601_chromophore_2', 'step_451_chromophore_15', 'step_801_chromophore_6', 'step_801_chromophore_14',
 'step_1101_chromophore_8', 'step_1_chromophore_18', 'step_601_chromophore_10', 'step_601_chromophore_21',
 'step_851_chromophore_24', 'step_1201_chromophore_1', 'step_1301_chromophore_3', 'step_351_chromophore_18',
 'step_1701_chromophore_8', 'step_851_chromophore_15', 'step_1351_chromophore_10', 'step_1351_chromophore_23',
 'step_601_chromophore_12', 'step_1301_chromophore_1', 'step_1651_chromophore_18', 'step_1401_chromophore_10',
 'step_151_chromophore_22', 'step_1401_chromophore_18', 'step_51_chromophore_7', 'step_1701_chromophore_3',
 'step_1801_chromophore_9', 'step_1501_chromophore_2', 'step_701_chromophore_1', 'step_1551_chromophore_8',
 'step_1201_chromophore_2', 'step_951_chromophore_8', 'step_101_chromophore_17', 'step_601_chromophore_14',
 'step_1251_chromophore_18', 'step_1601_chromophore_12', 'step_701_chromophore_13', 'step_1251_chromophore_10',
 'step_1101_chromophore_13', 'step_1301_chromophore_22', 'step_551_chromophore_17', 'step_951_chromophore_15',
 'step_1051_chromophore_12', 'step_201_chromophore_4', 'step_1101_chromophore_2', 'step_1151_chromophore_7',
 'step_101_chromophore_22', 'step_401_chromophore_20', 'step_1301_chromophore_11', 'step_701_chromophore_6',
 'step_1651_chromophore_7', 'step_51_chromophore_25', 'step_1401_chromophore_15', 'step_1051_chromophore_6',
 'step_751_chromophore_17', 'step_1701_chromophore_11', 'step_651_chromophore_5', 'step_51_chromophore_22',
 'step_1401_chromophore_21', 'step_101_chromophore_5', 'step_1_chromophore_26', 'step_101_chromophore_25',
 'step_1601_chromophore_20', 'step_301_chromophore_7', 'step_1801_chromophore_4', 'step_1051_chromophore_9',
 'step_251_chromophore_20', 'step_251_chromophore_14', 'step_501_chromophore_4', 'step_751_chromophore_10',
 'step_1001_chromophore_19', 'step_1_chromophore_27', 'step_1801_chromophore_14', 'step_51_chromophore_8',
 'step_401_chromophore_27', 'step_951_chromophore_26', 'step_1851_chromophore_9', 'step_1201_chromophore_3',
 'step_1001_chromophore_20', 'step_1351_chromophore_7', 'step_1501_chromophore_5', 'step_1301_chromophore_16',
 'step_1301_chromophore_20', 'step_651_chromophore_1', 'step_1251_chromophore_13', 'step_1051_chromophore_5',
 'step_1351_chromophore_8', 'step_951_chromophore_13', 'step_1351_chromophore_20', 'step_901_chromophore_16',
 'step_1701_chromophore_4', 'step_551_chromophore_21', 'step_751_chromophore_12', 'step_501_chromophore_2',
 'step_801_chromophore_9', 'step_1051_chromophore_15', 'step_1001_chromophore_21', 'step_51_chromophore_5',
 'step_601_chromophore_24', 'step_301_chromophore_10', 'step_651_chromophore_10', 'step_401_chromophore_25',
 'step_1451_chromophore_10', 'step_1701_chromophore_21', 'step_1701_chromophore_24', 'step_301_chromophore_19',
 'step_1251_chromophore_15', 'step_601_chromophore_11', 'step_151_chromophore_17', 'step_301_chromophore_20',
 'step_851_chromophore_4', 'step_1051_chromophore_7', 'step_351_chromophore_22', 'step_151_chromophore_14',
 'step_1401_chromophore_12', 'step_1751_chromophore_9', 'step_751_chromophore_16', 'step_1101_chromophore_18',
 'step_601_chromophore_27', 'step_951_chromophore_6', 'step_1_chromophore_9', 'step_1251_chromophore_19',
 'step_1551_chromophore_6', 'step_251_chromophore_5', 'step_901_chromophore_17', 'step_601_chromophore_1',
 'step_151_chromophore_13', 'step_251_chromophore_23', 'step_1451_chromophore_16', 'step_901_chromophore_11',
 'step_451_chromophore_23', 'step_651_chromophore_22', 'step_1701_chromophore_5', 'step_1851_chromophore_15',
 'step_501_chromophore_20', 'step_401_chromophore_24', 'step_401_chromophore_18', 'step_1801_chromophore_2',
 'step_601_chromophore_19', 'step_1101_chromophore_1', 'step_1051_chromophore_21', 'step_851_chromophore_26',
 'step_401_chromophore_1', 'step_551_chromophore_18', 'step_501_chromophore_13', 'step_451_chromophore_6',
 'step_151_chromophore_6', 'step_1601_chromophore_22', 'step_1151_chromophore_19', 'step_1451_chromophore_5',
 'step_401_chromophore_23', 'step_651_chromophore_20', 'step_1501_chromophore_12', 'step_201_chromophore_13',
 'step_101_chromophore_9', 'step_1201_chromophore_14', 'step_901_chromophore_1', 'step_1851_chromophore_22',
 'step_951_chromophore_2', 'step_1701_chromophore_9', 'step_351_chromophore_14', 'step_201_chromophore_26',
 'step_901_chromophore_18', 'step_601_chromophore_20', 'step_51_chromophore_9', 'step_151_chromophore_5',
 'step_1051_chromophore_1', 'step_101_chromophore_27', 'step_101_chromophore_15', 'step_1201_chromophore_13',
 'step_801_chromophore_23', 'step_1401_chromophore_2', 'step_451_chromophore_11', 'step_201_chromophore_11',
 'step_1201_chromophore_19', 'step_751_chromophore_2', 'step_1651_chromophore_3', 'step_851_chromophore_2',
 'step_551_chromophore_25', 'step_1551_chromophore_5', 'step_1751_chromophore_3', 'step_451_chromophore_12',
 'step_1451_chromophore_18', 'step_901_chromophore_2', 'step_1501_chromophore_20', 'step_951_chromophore_27',
 'step_1251_chromophore_27', 'step_801_chromophore_11', 'step_1751_chromophore_26', 'step_1001_chromophore_14',
 'step_1851_chromophore_16', 'step_651_chromophore_13', 'step_1751_chromophore_7', 'step_701_chromophore_16',
 'step_1151_chromophore_26', 'step_51_chromophore_20', 'step_301_chromophore_26', 'step_1501_chromophore_21',
 'step_651_chromophore_6', 'step_351_chromophore_16', 'step_1551_chromophore_16', 'step_851_chromophore_23',
 'step_751_chromophore_7', 'step_801_chromophore_25', 'step_551_chromophore_7', 'step_751_chromophore_3',
 'step_951_chromophore_22', 'step_51_chromophore_6', 'step_1101_chromophore_12', 'step_1851_chromophore_24',
 'step_1801_chromophore_23', 'step_1601_chromophore_9', 'step_1301_chromophore_26', 'step_1151_chromophore_18',
 'step_451_chromophore_14', 'step_1501_chromophore_3', 'step_351_chromophore_12', 'step_851_chromophore_12',
 'step_1051_chromophore_3', 'step_1451_chromophore_7', 'step_801_chromophore_24', 'step_101_chromophore_26',
 'step_701_chromophore_17', 'step_1651_chromophore_17', 'step_651_chromophore_16']

with open("validation_set.txt", 'w') as validation_file:
	for x in validation_set:
		print(x, file=validation_file)

test_set = [x for x in list(ref_data.keys()) if x not in validation_set]

with open("test_set.txt", 'w') as test_file:
	for x in test_set:
		print(x, file=test_file)

len_test_set = len(test_set)
print(f"# points in test set: {len_test_set}")

test_set_energies = np.array([ref_data[i]["energy"] * 27.2114 for i in test_set])

test_set_stddev = np.std(test_set_energies)


count = 0

training_set = random.sample(test_set, k=100)
training_set_energies = np.array([ref_data[i]["energy"] * 27.2114 for i in training_set])
while abs(np.std(training_set_energies) - test_set_stddev) > 1e-5:
	print(count)
	count += 1
	training_set = random.sample(test_set, k=100)
	training_set_energies = np.array([ref_data[i]["energy"] * 27.2114 for i in training_set])

with open("training_set.txt", 'w') as training_file:
	for x in training_set:
		print(x, file=training_file)



print(f"test set std. dev.: {test_set_stddev}")

samples_sizes = list(range(20, 520, 20))

subset_stddevs = [[] for i in samples_sizes]

for enum, k in enumerate(samples_sizes):
	for run in range(2000):
		subset = random.sample(test_set, k=k)
		subset_energies = np.array([ref_data[i]["energy"] * 27.2114 for i in subset])
		subset_stddev = np.std(subset_energies)
		subset_stddevs[enum].append(subset_stddev)

		#subset_stddev_error = 100 * (np.std(subset_energies) - test_set_stddev)/test_set_stddev
		#subset_stddev_errors[enum].append(subset_stddev_error)

fig, ax = plt.subplots()

#plt.violinplot(subset_variances, positions=list(samples_sizes), widths=5, showmeans=False, showmedians=True)
plt.boxplot(subset_stddevs, positions=list(samples_sizes), widths=5, showfliers=False)
ax.hlines([test_set_stddev], xmin=0, xmax=500, color='red', linestyle='--')

plt.title("Excitation energy standard deviation of training sets compared to the test set")
ax.set_xlabel("size of training set")
ax.set_ylabel("stddev in excitation energy / eV")

plt.tight_layout()

plt.show()













