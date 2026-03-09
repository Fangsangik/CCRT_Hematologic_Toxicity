"""
Patients Dataset 구성 시각화 + Project 구조 ERD
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 한글 폰트 설정
# ============================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Patients Dataset 구성 시각화
# ============================================================
def plot_patient_dataset_overview():
    """200명 환자 데이터셋의 전체 구성을 한눈에 보여주는 대시보드"""

    df = pd.read_csv('/home/user/CCRT_Hematologic_Toxicity/data/raw/emr_synthetic/emr_patients.csv')
    emr = pd.read_csv('/home/user/CCRT_Hematologic_Toxicity/data/processed/emr_processed.csv')

    fig = plt.figure(figsize=(28, 20))
    fig.suptitle('CCRT Hematologic Toxicity - Patient Dataset Overview (N=200)',
                 fontsize=22, fontweight='bold', y=0.98)

    # Grid layout: 4 rows x 4 cols
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35,
                          left=0.06, right=0.96, top=0.93, bottom=0.04)

    colors = {
        'primary': '#2196F3', 'secondary': '#FF9800', 'accent': '#4CAF50',
        'danger': '#F44336', 'purple': '#9C27B0', 'teal': '#009688',
        'male': '#42A5F5', 'female': '#EF5350',
    }

    # --- (1) Age Distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['age'], bins=15, color=colors['primary'], alpha=0.8, edgecolor='white', linewidth=0.8)
    ax1.axvline(df['age'].mean(), color=colors['danger'], linestyle='--', linewidth=2,
                label=f'Mean: {df["age"].mean():.1f}')
    ax1.axvline(df['age'].median(), color=colors['secondary'], linestyle='-.', linewidth=2,
                label=f'Median: {df["age"].median():.1f}')
    ax1.set_title('Age Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Count')
    ax1.legend(fontsize=9)

    # --- (2) Sex Distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    sex_counts = df['sex'].value_counts()
    wedges, texts, autotexts = ax2.pie(
        sex_counts.values, labels=None, autopct='%1.1f%%',
        colors=[colors['male'], colors['female']], startangle=90,
        pctdistance=0.75, textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    ax2.legend([f'Male (n={sex_counts.get("M", 0)})', f'Female (n={sex_counts.get("F", 0)})'],
              loc='lower center', fontsize=10)
    ax2.set_title('Sex Distribution', fontsize=13, fontweight='bold')

    # --- (3) ECOG PS ---
    ax3 = fig.add_subplot(gs[0, 2])
    ecog_counts = df['ecog_ps'].value_counts().sort_index()
    bars = ax3.bar(ecog_counts.index.astype(str), ecog_counts.values,
                   color=[colors['accent'], colors['primary'], colors['secondary']][:len(ecog_counts)],
                   edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, ecog_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold', fontsize=11)
    ax3.set_title('ECOG Performance Status', fontsize=13, fontweight='bold')
    ax3.set_xlabel('ECOG PS')
    ax3.set_ylabel('Count')

    # --- (4) BMI Distribution ---
    ax4 = fig.add_subplot(gs[0, 3])
    bmi_cats = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 50],
                      labels=['Underweight\n(<18.5)', 'Normal\n(18.5-25)',
                              'Overweight\n(25-30)', 'Obese\n(>30)'])
    bmi_counts = bmi_cats.value_counts().reindex(['Underweight\n(<18.5)', 'Normal\n(18.5-25)',
                                                   'Overweight\n(25-30)', 'Obese\n(>30)'])
    bar_colors = ['#FFCA28', colors['accent'], colors['secondary'], colors['danger']]
    bars = ax4.bar(range(len(bmi_counts)), bmi_counts.values, color=bar_colors, edgecolor='white')
    ax4.set_xticks(range(len(bmi_counts)))
    ax4.set_xticklabels(bmi_counts.index, fontsize=9)
    for bar, val in zip(bars, bmi_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold', fontsize=11)
    ax4.set_title('BMI Categories', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Count')

    # --- (5) Stage Distribution ---
    ax5 = fig.add_subplot(gs[1, 0])
    stage_counts = df['stage'].value_counts().sort_index()
    stage_colors = ['#66BB6A', '#FFA726', '#EF5350']
    bars = ax5.barh(stage_counts.index, stage_counts.values, color=stage_colors[:len(stage_counts)],
                    edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, stage_counts.values):
        ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val} ({val/len(df)*100:.0f}%)', va='center', fontweight='bold', fontsize=10)
    ax5.set_title('Cancer Stage (AJCC)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Count')

    # --- (6) T Stage ---
    ax6 = fig.add_subplot(gs[1, 1])
    t_counts = df['t_stage'].value_counts().sort_index()
    t_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(t_counts)))
    bars = ax6.bar(t_counts.index, t_counts.values, color=t_colors, edgecolor='white')
    for bar, val in zip(bars, t_counts.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold', fontsize=10)
    ax6.set_title('T Stage', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Count')

    # --- (7) N Stage ---
    ax7 = fig.add_subplot(gs[1, 2])
    n_counts = df['n_stage'].value_counts().sort_index()
    n_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(n_counts)))
    bars = ax7.bar(n_counts.index, n_counts.values, color=n_colors, edgecolor='white')
    for bar, val in zip(bars, n_counts.values):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold', fontsize=10)
    ax7.set_title('N Stage', fontsize=13, fontweight='bold')
    ax7.set_ylabel('Count')

    # --- (8) Chemo Regimen ---
    ax8 = fig.add_subplot(gs[1, 3])
    chemo_counts = df['chemo_regimen'].value_counts()
    chemo_colors = plt.cm.Set2(np.linspace(0, 1, len(chemo_counts)))
    wedges, texts, autotexts = ax8.pie(
        chemo_counts.values, labels=None, autopct='%1.0f%%',
        colors=chemo_colors, startangle=90, pctdistance=0.8,
        textprops={'fontsize': 9, 'fontweight': 'bold'}
    )
    ax8.legend([f'{k} (n={v})' for k, v in chemo_counts.items()],
              loc='lower center', fontsize=8, ncol=2)
    ax8.set_title('Chemotherapy Regimen', fontsize=13, fontweight='bold')

    # --- (9) RT Dose Distribution ---
    ax9 = fig.add_subplot(gs[2, 0])
    rt_counts = df['rt_total_dose'].value_counts().sort_index()
    bars = ax9.bar(rt_counts.index.astype(str), rt_counts.values,
                   color=colors['teal'], edgecolor='white')
    for bar, val in zip(bars, rt_counts.values):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontweight='bold', fontsize=10)
    ax9.set_title('RT Total Dose (Gy)', fontsize=13, fontweight='bold')
    ax9.set_ylabel('Count')

    # --- (10) Chemo Cycles ---
    ax10 = fig.add_subplot(gs[2, 1])
    cycle_counts = df['chemo_cycles'].value_counts().sort_index()
    bars = ax10.bar(cycle_counts.index.astype(str), cycle_counts.values,
                    color=colors['purple'], alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, cycle_counts.values):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', fontweight='bold', fontsize=10)
    ax10.set_title('Chemotherapy Cycles', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Count')

    # --- (11) Lab Values (Creatinine & Albumin) ---
    ax11 = fig.add_subplot(gs[2, 2])
    bp = ax11.boxplot([df['creatinine'].dropna(), df['albumin'].dropna()],
                      labels=['Creatinine\n(mg/dL)', 'Albumin\n(g/dL)'],
                      patch_artist=True, widths=0.5,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#81D4FA')
    bp['boxes'][1].set_facecolor('#A5D6A7')
    ax11.set_title('Baseline Lab Values', fontsize=13, fontweight='bold')

    # --- (12) Toxicity Outcome (Grade 3+) ---
    ax12 = fig.add_subplot(gs[2, 3])
    tox_types = ['grade3_neutropenia', 'grade3_anemia', 'grade3_thrombocytopenia',
                 'grade3_leukopenia', 'grade3_lymphopenia']
    tox_labels = ['Neutropenia', 'Anemia', 'Thrombocyto-\npenia', 'Leukopenia', 'Lymphopenia']
    tox_rates = [emr[t].mean() * 100 for t in tox_types]
    tox_colors = ['#F44336', '#FF9800', '#9C27B0', '#2196F3', '#4CAF50']
    bars = ax12.barh(tox_labels, tox_rates, color=tox_colors, edgecolor='white', height=0.6)
    for bar, rate in zip(bars, tox_rates):
        ax12.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{rate:.1f}%', va='center', fontweight='bold', fontsize=10)
    ax12.set_title('Grade 3+ Toxicity Rate', fontsize=13, fontweight='bold')
    ax12.set_xlabel('Incidence (%)')
    ax12.set_xlim(0, max(tox_rates) + 15)

    # --- (13) CBC Baseline Distribution (Week 0) ---
    ax13 = fig.add_subplot(gs[3, :2])
    cbc_cols = ['WBC_week0', 'ANC_week0', 'ALC_week0', 'AMC_week0', 'PLT_week0', 'Hb_week0']
    cbc_labels = ['WBC\n(10^3/uL)', 'ANC\n(10^3/uL)', 'ALC\n(10^3/uL)',
                  'AMC\n(10^3/uL)', 'PLT\n(10^3/uL)', 'Hb\n(g/dL)']

    # Normalize for comparison
    cbc_data = emr[cbc_cols]
    cbc_norm = (cbc_data - cbc_data.min()) / (cbc_data.max() - cbc_data.min())

    bp2 = ax13.boxplot([cbc_norm[c].dropna() for c in cbc_cols],
                       labels=cbc_labels, patch_artist=True, widths=0.5,
                       medianprops=dict(color='red', linewidth=2))
    cbc_colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#EC407A', '#26A69A']
    for patch, color in zip(bp2['boxes'], cbc_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax13.set_title('Baseline CBC Distribution (Week 0, Normalized)', fontsize=13, fontweight='bold')
    ax13.set_ylabel('Normalized Value')

    # Add actual value ranges as text
    for i, col in enumerate(cbc_cols):
        vals = emr[col]
        ax13.text(i + 1, -0.12, f'[{vals.min():.1f}-{vals.max():.1f}]',
                 ha='center', fontsize=8, color='gray')

    # --- (14) Data Summary Table ---
    ax14 = fig.add_subplot(gs[3, 2:])
    ax14.axis('off')

    summary_data = [
        ['Total Patients', f'{len(df)}'],
        ['Age Range', f'{df["age"].min()}-{df["age"].max()} (mean: {df["age"].mean():.1f})'],
        ['Male / Female', f'{(df["sex"]=="M").sum()} / {(df["sex"]=="F").sum()}'],
        ['BMI Range', f'{df["bmi"].min():.1f}-{df["bmi"].max():.1f}'],
        ['Stage IIIA/IIIB/IIIC', f'{(df["stage"]=="IIIA").sum()} / {(df["stage"]=="IIIB").sum()} / {(df["stage"]=="IIIC").sum()}'],
        ['RT Dose (Gy)', f'{df["rt_total_dose"].min():.0f}-{df["rt_total_dose"].max():.0f}'],
        ['Chemo Regimens', f'{df["chemo_regimen"].nunique()} types'],
        ['CBC Timepoints', 'Week 0-7 (8 weeks)'],
        ['CBC Features', 'WBC, ANC, ALC, AMC, PLT, Hb'],
        ['Target (Primary)', 'Grade 3+ Neutropenia'],
        ['Positive Rate', f'{emr["grade3_neutropenia"].mean()*100:.1f}% (n={emr["grade3_neutropenia"].sum()})'],
    ]

    table = ax14.table(cellText=summary_data,
                       colLabels=['Variable', 'Value'],
                       cellLoc='left', loc='center',
                       colWidths=[0.35, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Style rows
    for i in range(1, len(summary_data) + 1):
        for j in range(2):
            table[i, j].set_facecolor('#ECEFF1' if i % 2 == 0 else 'white')

    ax14.set_title('Dataset Summary', fontsize=13, fontweight='bold', pad=20)

    plt.savefig('/home/user/CCRT_Hematologic_Toxicity/outputs/figures/patient_dataset_overview.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Patient dataset overview saved!")


# ============================================================
# 2. Project Structure ERD
# ============================================================
def plot_project_erd():
    """Project architecture as an ERD-style diagram"""

    fig, ax = plt.subplots(1, 1, figsize=(32, 22))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    ax.set_title('CCRT Hematologic Toxicity Prediction\nProject Architecture & Data Flow ERD',
                fontsize=24, fontweight='bold', pad=20, color='#212121')

    # ---- Helper functions ----
    def draw_entity(ax, x, y, w, h, title, fields, title_color='#1565C0',
                    body_color='#E3F2FD', border_color='#1565C0'):
        """Draw an ERD entity box"""
        # Title bar
        title_box = FancyBboxPatch((x, y + h - 2.2), w, 2.2,
                                    boxstyle="round,pad=0.15",
                                    facecolor=title_color, edgecolor=border_color, linewidth=2)
        ax.add_patch(title_box)
        ax.text(x + w/2, y + h - 1.1, title, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')

        # Body
        body_box = FancyBboxPatch((x, y), w, h - 2.2,
                                   boxstyle="round,pad=0.15",
                                   facecolor=body_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(body_box)

        # Fields
        line_height = (h - 2.8) / max(len(fields), 1)
        for i, field in enumerate(fields):
            fy = y + h - 2.8 - (i + 0.5) * line_height
            if fy > y + 0.3:
                if field.startswith('PK') or field.startswith('FK'):
                    ax.text(x + 0.5, fy, field, fontsize=8.5, va='center',
                           fontweight='bold', color='#B71C1C', family='monospace')
                elif field.startswith('--'):
                    ax.text(x + w/2, fy, field.strip('-'), fontsize=8, va='center',
                           ha='center', color='#757575', style='italic')
                else:
                    ax.text(x + 0.5, fy, field, fontsize=8.5, va='center',
                           color='#333333', family='monospace')

    def draw_arrow(ax, x1, y1, x2, y2, label='', color='#546E7A', style='->', lw=1.8):
        """Draw a relationship arrow"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                  connectionstyle='arc3,rad=0.1'))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.5, label, fontsize=8, ha='center', va='center',
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=color, alpha=0.9))

    def draw_module_box(ax, x, y, w, h, title, color='#E8EAF6', border='#3F51B5'):
        """Draw a module grouping box"""
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                             facecolor=color, edgecolor=border, linewidth=2, alpha=0.3, linestyle='--')
        ax.add_patch(box)
        ax.text(x + 0.5, y + h - 0.5, title, fontsize=10, fontweight='bold',
               color=border, style='italic')

    # ============================================================
    # Layer 1: DATA SOURCE (Top)
    # ============================================================
    draw_module_box(ax, 0.5, 55.5, 99, 12, 'DATA LAYER', '#FFF3E0', '#E65100')

    # EMR Patients Table
    draw_entity(ax, 2, 57, 18, 10, 'emr_patients.csv', [
        'PK patient_id  (VARCHAR)',
        'age            (INT)',
        'sex            (M/F)',
        'bmi            (FLOAT)',
        'ecog_ps        (INT 0-4)',
        'stage          (IIIA/B/C)',
        't_stage        (T1-T4)',
        'n_stage        (N0-N3)',
        'creatinine     (FLOAT)',
        'albumin        (FLOAT)',
        'rt_start_date  (DATE)',
        'rt_total_dose  (FLOAT)',
        'rt_fraction_dose (FLOAT)',
        'chemo_regimen  (VARCHAR)',
        'chemo_dose     (FLOAT)',
        'chemo_cycles   (INT)',
    ], '#E65100', '#FFF3E0', '#E65100')

    # EMR CBC Results Table
    draw_entity(ax, 25, 57, 18, 10, 'emr_cbc_results.csv', [
        'FK patient_id  (VARCHAR)',
        'exam_date      (DATE)',
        'WBC            (FLOAT)',
        'ANC            (FLOAT)',
        'ALC            (FLOAT)',
        'AMC            (FLOAT)',
        'PLT            (FLOAT)',
        'Hb             (FLOAT)',
        '-- Long format --',
        '-- ~600 rows (200x3) --',
    ], '#E65100', '#FFF3E0', '#E65100')

    # Config
    draw_entity(ax, 48, 59, 16, 8, 'config.py', [
        'PathConfig',
        'DataConfig',
        'LSTMConfig',
        'XGBoostConfig',
        'LightGBMConfig',
        'LogRegConfig',
        'TrainConfig',
        '-- Central settings --',
    ], '#37474F', '#ECEFF1', '#37474F')

    # CTCAE Criteria
    draw_entity(ax, 68, 59, 14, 8, 'ctcae_v5_criteria.json', [
        'Neutropenia grades',
        '  G3: ANC < 1.0',
        '  G4: ANC < 0.5',
        'Anemia grades',
        '  G3: Hb < 8.0',
        'Thrombocytopenia',
        '  G3: PLT < 50',
    ], '#880E4F', '#FCE4EC', '#880E4F')

    # Processed Data
    draw_entity(ax, 85, 59, 14, 8, 'emr_processed.csv', [
        'PK patient_id',
        'CBC_week0..7 (48 cols)',
        'grade_week0..7 (40)',
        'max_grade_* (5)',
        'grade3_* (5 targets)',
        'clinical features',
        '-- Wide format --',
        '-- 200 x 108 cols --',
    ], '#1B5E20', '#E8F5E9', '#1B5E20')

    # ============================================================
    # Layer 2: PREPROCESSING (Middle-top)
    # ============================================================
    draw_module_box(ax, 0.5, 42, 99, 12, 'PREPROCESSING LAYER (src/data/)', '#E8EAF6', '#283593')

    draw_entity(ax, 2, 43.5, 16, 9.5, 'preprocessing.py', [
        'EMRPreprocessor',
        '  assign_treatment_week()',
        '  convert_long_to_wide()',
        '  calculate_ctcae_grades()',
        '  interpolate_missing()',
        '  run_full_pipeline()',
        '-- Date->Week mapping --',
        '-- CTCAE Grade calc --',
    ], '#283593', '#E8EAF6', '#283593')

    draw_entity(ax, 21, 43.5, 16, 9.5, 'data_loader.py', [
        'CCRTDataLoader',
        '  load_data()',
        '  explore_data()',
        '  handle_missing()',
        '  handle_outliers()',
        '  split_data()',
        '-- Stratified split --',
        '-- Median imputation --',
    ], '#283593', '#E8EAF6', '#283593')

    draw_entity(ax, 40, 43.5, 16, 9.5, 'feature_engineer.py', [
        'FeatureEngineer',
        '  create_temporal_features()',
        '  encode_categorical()',
        '  select_features()',
        '  create_lstm_sequences()',
        '-- delta, slope, cv --',
        '-- nadir, pct_change --',
        '-- AMC/ANC ratio --',
    ], '#283593', '#E8EAF6', '#283593')

    draw_entity(ax, 59, 43.5, 16, 9.5, 'dataset.py', [
        'CCRTDataset(Dataset)',
        '  __getitem__()',
        '  mask_augmentation()',
        '  create_dataloaders()',
        '-- PyTorch Dataset --',
        '-- Week masking 80% --',
        '-- Sequence + Static --',
    ], '#283593', '#E8EAF6', '#283593')

    # Feature detail box
    draw_entity(ax, 78, 43.5, 21, 9.5, 'Feature Schema', [
        'Baseline (9): age,sex,bmi,...',
        'Treatment (5): rt_dose,chemo,...',
        'CBC raw (18): WBC/ANC/..._wk0-2',
        'CBC derived (52):',
        '  delta_w0w1, delta_w1w2',
        '  slope, pct_change, cv',
        '  nadir, AMC_ANC_ratio',
        'Targets: grade3_neutropenia',
    ], '#4A148C', '#F3E5F5', '#4A148C')

    # ============================================================
    # Layer 3: MODELS
    # ============================================================
    draw_module_box(ax, 0.5, 22, 99, 18.5, 'MODEL LAYER (src/models/)', '#E0F2F1', '#004D40')

    draw_entity(ax, 2, 23.5, 18, 9, 'lstm_model.py', [
        'LSTMPredictor(nn.Module)',
        '  LSTM encoder (2 layers)',
        '  TemporalAttention',
        '  FC baseline encoder',
        '  Combined classifier',
        '-- hidden=64, drop=0.3 --',
        '-- Attention weights --',
    ], '#004D40', '#E0F2F1', '#004D40')

    draw_entity(ax, 23, 23.5, 17, 9, 'forecaster.py', [
        'CBCForecasterV2',
        '  encode: Week 0-2',
        '  decode: Week 3-7',
        '  predict future CBC',
        '-- Seq2Seq style --',
        '-- 6 CBC features --',
    ], '#004D40', '#E0F2F1', '#004D40')

    draw_entity(ax, 43, 23.5, 17, 9, 'baseline_models.py', [
        'XGBoostModel',
        'LightGBMModel',
        'LogisticRegressionModel',
        '  train() / predict()',
        '  feature_importance()',
        '-- class_weight=balanced --',
        '-- SMOTE oversampling --',
    ], '#004D40', '#E0F2F1', '#004D40')

    draw_entity(ax, 63, 23.5, 17, 9, 'trainer.py', [
        'LSTMTrainer',
        '  train_epoch()',
        '  validate()',
        'EarlyStopping',
        'CrossValidator (5-fold)',
        '-- BCE + class weight --',
        '-- StratifiedKFold --',
    ], '#004D40', '#E0F2F1', '#004D40')

    # Experiment modes
    draw_entity(ax, 2, 33.5, 22, 6, 'main.py (Entry Points)', [
        'mode=demo    : synthetic data',
        'mode=train   : preprocessed CSV',
        'mode=emr     : raw EMR pipeline',
        'mode=template: data templates',
        '-- 2 experiments --',
        '  baseline_only vs baseline_cbc',
    ], '#BF360C', '#FBE9E7', '#BF360C')

    draw_entity(ax, 27, 33.5, 18, 6, 'lstm_forecast.py', [
        'Forecast-then-Classify',
        '  1) Train forecaster',
        '  2) Predict Week 3-7',
        '  3) Classify toxicity',
        '-- 2-stage pipeline --',
    ], '#BF360C', '#FBE9E7', '#BF360C')

    draw_entity(ax, 48, 33.5, 16, 6, 'lstm_e2e.py', [
        'End-to-End Pipeline',
        '  Single model',
        '  Week 0-2 input',
        '  Direct prediction',
    ], '#BF360C', '#FBE9E7', '#BF360C')

    draw_entity(ax, 67, 33.5, 16, 6, 'external_validation.py', [
        'External Validation',
        '  Load saved models',
        '  Evaluate on new data',
        '  Compare all models',
    ], '#BF360C', '#FBE9E7', '#BF360C')

    # Models summary
    draw_entity(ax, 83, 23.5, 16, 16, '7 Model Configs', [
        '-- baseline_only --',
        '1. XGBoost',
        '2. LightGBM',
        '3. Logistic Reg',
        '',
        '-- baseline_cbc --',
        '4. XGBoost + CBC',
        '5. LightGBM + CBC',
        '6. LogReg + CBC',
        '7. LSTM + Attention',
        '',
        '-- Extended --',
        '8. LSTM Forecaster',
        '9. LSTM E2E',
        '10. LSTM Oracle',
    ], '#311B92', '#EDE7F6', '#311B92')

    # ============================================================
    # Layer 4: EVALUATION (Bottom)
    # ============================================================
    draw_module_box(ax, 0.5, 1, 99, 19.5, 'EVALUATION & OUTPUT LAYER (src/evaluation/)', '#FFF8E1', '#F57F17')

    draw_entity(ax, 2, 2.5, 18, 9, 'metrics.py', [
        'compute_all_metrics()',
        '  AUROC, AUPRC',
        '  Sensitivity, Specificity',
        '  PPV, NPV, F1',
        'bootstrap_ci()',
        'compare_models()',
        'compute_incremental_value()',
    ], '#F57F17', '#FFF8E1', '#F57F17')

    draw_entity(ax, 23, 2.5, 18, 9, 'visualization.py', [
        'plot_roc_curves()',
        'plot_pr_curves()',
        'plot_confusion_matrix()',
        'plot_feature_importance()',
        'plot_cbc_timeseries()',
        'plot_model_comparison()',
        'plot_training_history()',
    ], '#F57F17', '#FFF8E1', '#F57F17')

    draw_entity(ax, 44, 2.5, 17, 9, 'helpers.py', [
        'setup_logging()',
        'set_seed(42)',
        'print_data_summary()',
        'save_results()',
        '-- Reproducibility --',
    ], '#F57F17', '#FFF8E1', '#F57F17')

    # Output files
    draw_entity(ax, 64, 2.5, 16, 9, 'outputs/figures/', [
        'cbc_timeseries_*.png',
        'model_comparison.png',
        'roc_comparison.png',
        'feature_importance_*.png',
        'lstm_training_history.png',
        'roc_external_validation.png',
        '-- 14 figures total --',
    ], '#E65100', '#FFF3E0', '#E65100')

    draw_entity(ax, 83, 2.5, 16, 9, 'outputs/logs/', [
        'config.yaml',
        'experiment_results.json',
        'external_validation.json',
        'train_*.log',
        '-- All metrics saved --',
        '-- Reproducible --',
    ], '#E65100', '#FFF3E0', '#E65100')

    # Cardinality labels on side
    draw_entity(ax, 64, 13, 16, 6.5, 'outputs/models/', [
        '*.pkl (sklearn models)',
        '*.pt  (PyTorch models)',
        'xgboost, lightgbm',
        'logistic_regression',
        'lstm_predictor',
    ], '#E65100', '#FFF3E0', '#E65100')

    draw_entity(ax, 83, 13, 16, 6.5, 'data/processed/', [
        'processed_data.csv (3 rows)',
        'emr_processed.csv (200)',
        'external_validation.csv',
        'forecast_external.csv',
        'e2e_external.csv',
    ], '#E65100', '#FFF3E0', '#E65100')

    # ============================================================
    # ARROWS (Data Flow)
    # ============================================================
    # Data sources -> Preprocessing
    draw_arrow(ax, 11, 57, 10, 53, '1:N', '#E65100', '->', 2)
    draw_arrow(ax, 34, 57, 29, 53, 'Long', '#E65100', '->', 2)

    # Preprocessing chain
    draw_arrow(ax, 18, 48, 21, 48, '', '#283593', '->', 1.5)
    draw_arrow(ax, 37, 48, 40, 48, '', '#283593', '->', 1.5)
    draw_arrow(ax, 56, 48, 59, 48, '', '#283593', '->', 1.5)
    draw_arrow(ax, 75, 48, 78, 48, '', '#283593', '->', 1.5)

    # Config -> everything
    draw_arrow(ax, 56, 59, 40, 53, 'config', '#546E7A', '->', 1)
    draw_arrow(ax, 56, 59, 59, 53, 'config', '#546E7A', '->', 1)

    # Preprocessing -> Models
    draw_arrow(ax, 10, 43.5, 10, 39.5, '', '#283593', '->', 2)
    draw_arrow(ax, 36, 43.5, 36, 39.5, '', '#283593', '->', 2)
    draw_arrow(ax, 67, 43.5, 67, 39.5, '', '#283593', '->', 2)

    # Entry points -> Models
    draw_arrow(ax, 13, 33.5, 11, 32.5, '', '#BF360C', '->', 1.5)
    draw_arrow(ax, 36, 33.5, 34, 32.5, '', '#BF360C', '->', 1.5)
    draw_arrow(ax, 56, 33.5, 54, 32.5, '', '#BF360C', '->', 1.5)

    # Models -> Evaluation
    draw_arrow(ax, 11, 23.5, 11, 11.5, '', '#004D40', '->', 2)
    draw_arrow(ax, 51, 23.5, 32, 11.5, '', '#004D40', '->', 2)
    draw_arrow(ax, 72, 23.5, 55, 11.5, '', '#004D40', '->', 2)

    # Evaluation -> Outputs
    draw_arrow(ax, 41, 7, 44, 7, '', '#F57F17', '->', 1.5)
    draw_arrow(ax, 61, 7, 64, 7, '', '#F57F17', '->', 1.5)
    draw_arrow(ax, 80, 7, 83, 7, '', '#F57F17', '->', 1.5)

    # CTCAE -> preprocessing
    draw_arrow(ax, 75, 59, 18, 53, 'CTCAE thresholds', '#880E4F', '->', 1.5)

    # To processed data
    draw_arrow(ax, 75, 48, 85, 59, 'output', '#1B5E20', '->', 1.5)

    plt.savefig('/home/user/CCRT_Hematologic_Toxicity/outputs/figures/project_erd.png',
                dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print("Project ERD saved!")


if __name__ == '__main__':
    plot_patient_dataset_overview()
    plot_project_erd()
    print("\nAll visualizations complete!")
    print("  1. outputs/figures/patient_dataset_overview.png")
    print("  2. outputs/figures/project_erd.png")
