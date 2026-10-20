# OPTIONAL: paste into a new notebook cell right after the EDA section (Section 6/7).
# These produce the extra readmission-rate-by-group figures quoted in the README.

print("Readmission rate (<30 days) by number of prior inpatient visits")
prior_bins = pd.cut(df['number_inpatient'], [-1, 0, 1, 2, 100], labels=['0', '1', '2', '3+'])
print((df.groupby(prior_bins)['readmitted_binary'].agg(['count', 'mean']) * [1, 100]).round(1))

print("\nReadmission rate by number of prior emergency visits")
er_bins = pd.cut(df['number_emergency'], [-1, 0, 1, 100], labels=['0', '1', '2+'])
print((df.groupby(er_bins)['readmitted_binary'].agg(['count', 'mean']) * [1, 100]).round(1))

print("\nReadmission rate by primary diagnosis category")
print((df.groupby('diag_1_category')['readmitted_binary'].agg(['count', 'mean']) * [1, 100])
      .round(1).sort_values('mean', ascending=False))

print("\nReadmission rate by length of stay")
los_bins = pd.cut(df['time_in_hospital'], [0, 3, 7, 14], labels=['1-3 days', '4-7 days', '8-14 days'])
print((df.groupby(los_bins)['readmitted_binary'].agg(['count', 'mean']) * [1, 100]).round(1))

print("\nUnique patients:", pd.read_csv('diabetes_data/diabetic_data.csv', usecols=['patient_nbr'])['patient_nbr'].nunique())
