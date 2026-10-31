import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np




########## bootstrap CI helpers #############
def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.95):
    scores = []
    n = len(y_true)
    for i in range(n_bootstrap):
        # Bootstrap resample with replacement
        indices = np.random.choice(range(n), size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        scores.append(metric_func(y_true_sample, y_pred_sample))
    # Compute confidence intervals
    lower_bound = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(scores, (alpha + (1 - alpha) / 2) * 100)
    return lower_bound, upper_bound

# Sensitivity / Specificity from the confusion matrix
def sensitivity_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def specificity_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
########## bootstrap CI helpers #############



# Read the labelled training spreadsheet
df = pd.read_excel('AI_dryeye_diagnosis.xlsx')

df_new = pd.DataFrame()

i = 0






# Dependent (label) and independent (features) variables — DE_new
y = df['gold_standard_OSDI_NIBUT']
# y = df['multi_class']
# Available columns for reference:
# gender age OSDI NIBUT_OD Logmr_od BCVA_OD IOP_mmHg_OD max_blink_interval blink_freq_per_min
# max_palpebral_fissure_height min_palpebral_fissure_height complete_blink_count
# incomplete_blink_count total_blink_count min_blink_duration mean_blink_duration
# max_eye_close_duration max_eye_open_duration
# independent_vars = ['gender', 'age', 'OSDI', 'NIBUT_OD', 'Logmr_od', 'BCVA_OD', 'IOP_mmHg_OD', 'max_blink_interval', 'blink_freq_per_min', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['gender', 'age', 'OSDI', 'Logmr_od', 'BCVA_OD', 'IOP_mmHg_OD', 'max_blink_interval', 'blink_freq_per_min', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['gender', 'age', 'NIBUT_OD', 'Logmr_od', 'BCVA_OD', 'IOP_mmHg_OD', 'max_blink_interval', 'blink_freq_per_min', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['NIBUT_OD']
# Full feature set including the eye-redness signal coming out of DE_RedEye/red_eye_detect.py
# Un-comment the next line and comment the active one to retrain with the redness feature.
# independent_vars = ['OSDI', 'max_blink_interval', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration', 'eye_redness_prob']
independent_vars = ['OSDI', 'max_blink_interval', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['max_blink_interval', 'blink_freq_per_min', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['gender', 'age', 'max_blink_interval', 'blink_freq_per_min', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count', 'total_blink_count', 'min_blink_duration', 'mean_blink_duration', 'max_eye_close_duration', 'max_eye_open_duration']
# independent_vars = ['OSDI', 'max_blink_interval', 'max_palpebral_fissure_height', 'min_palpebral_fissure_height', 'complete_blink_count', 'incomplete_blink_count']


X = df[independent_vars]



# Build the XGBoost classifier
xgb_model = XGBClassifier(n_estimators=100, random_state=6)


# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)

# Cross-validated predictions and probabilities
y_pred = cross_val_predict(xgb_model, X, y, cv=cv)
y_pred_prob = cross_val_predict(xgb_model, X, y, cv=cv, method='predict_proba')[:, 1]

# Cross-validated y_true / y_pred for downstream metrics
y_true = y

########## Bootstrap 95% CIs #############
auc_ci = bootstrap_ci(y_true, y_pred_prob, lambda yt, yp: roc_auc_score(yt, yp))
acc_ci = bootstrap_ci(y_true, y_pred, lambda yt, yp: accuracy_score(yt, yp))
f1_ci = bootstrap_ci(y_true, y_pred, lambda yt, yp: f1_score(yt, yp))

sens_ci = bootstrap_ci(y_true, y_pred, sensitivity_fn)
spec_ci = bootstrap_ci(y_true, y_pred, specificity_fn)
########## Bootstrap 95% CIs #############

# Evaluation
# Accuracy
df_new.at[i, 'Class'] = 'DE'
print(f"Accuracy: {accuracy_score(y, y_pred)} (95%CI:{acc_ci[0]:.3f},{acc_ci[1]:.3f})")
df_new.at[i, "Accuracy"] = f"{accuracy_score(y, y_pred):.3f} (95% CI: {acc_ci[0]:.3f}, {acc_ci[1]:.3f})"
# Sensitivity (recall)
print("Sensitivity:", recall_score(y, y_pred, pos_label=1))
df_new.at[i, "Sensitivity"] = f"{recall_score(y, y_pred):.3f} (95% CI: {sens_ci[0]:.3f}, {sens_ci[1]:.3f})"
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
# Confusion-matrix derived counts
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
# Specificity
specificity = tn / (tn + fp)
print("Specificity:", specificity)
df_new.at[i, "Specificity"] = f"{specificity:.3f} (95% CI: {spec_ci[0]:.3f}, {spec_ci[1]:.3f})"
# F1 score
print("F1 score:", f1_score(y, y_pred))
df_new.at[i, "F1 score"] = f"{f1_score(y, y_pred):.3f} (95% CI: {f1_ci[0]:.3f}, {f1_ci[1]:.3f})"

print("Classification Report:\n", classification_report(y, y_pred))




# Refit on the whole dataset to expose feature importances; keep the top-5
xgb_model.fit(X, y)
importances = xgb_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': independent_vars, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False).head(5)
print("Top 5 Feature Importances:\n", feature_importances)



# Plot the top-5 feature importances
plt.figure(figsize=(4, 4))

bars = plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Diagnose top 5 Feature Importance from XGBoost Model')
plt.gca().invert_yaxis()  # largest importance on top

# Annotate each bar with its weight value
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')

# plt.savefig("importance_results.jpg")

# Compute and print AUC
auc = roc_auc_score(y, y_pred_prob)
print("AUC:", auc)
df_new.at[i, "AUC"] = f"{auc:.3f} (95% CI: {auc_ci[0]:.3f}, {auc_ci[1]:.3f})"

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)



i += 1




plt.figure()
plt.plot(fpr, tpr, color='blue', label='(AUC=%0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Diagnose internal 5-fold cross validation')
plt.legend(loc="lower right")
plt.savefig('result.jpg')

df_new.to_excel('XGB_with_95CI_result_diagnose.xlsx')

# Persist the trained model and the ordered feature-name list for inference
xgb_model.save_model('xgb_model.json')

import pickle
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(independent_vars, f)
