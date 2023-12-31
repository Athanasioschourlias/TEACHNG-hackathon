# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {

    "data": {
        "path_normal": "data/unsw-nb15/short_attack_normal/normal_short.csv",
        "path_anomaly": "data/unsw-nb15/short_attack_normal/attack_short.csv",
        "verification_set": "data/verification/UNSW-NB15_1.csv",
        "ground_truth_cols": ['label'],  # a list with names of columns or None
        "features": ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl",
                    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
                    "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
                    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
                    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"],
        "data_types": ["Float32", "category", "category", "category", "Int64", "Int64", "Int64", "Int64",
                       "Int64", "Int64", "Float32", "Float32", "Int64", "Int64", "Float32", "Float32",
                       "Float32", "Float32", "Int64", "Int64", "Int64", "Int64", "Float32", "Float32", "Float32",
                       "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64", "Int64",
                       "category", "Int64", "Int64", "Int64", "Int64", "category"],
        "n_rows": None
    },
    "tune": {
        "tuning": True,
        "max_evals": 50
    },
    "train": {
        "seq_time_steps": 6,
        "batch_size": 25,
        "epochs": 100,
        "val_subsplits": 0.1,
        "units": [102, 102],
        "dropout_rate": [0.2, 0.2],
        "loss": "mse",
        "early_stopping_rounds": 5,
        "learning_rate": 0.0032265
    },
    "model": {
        "model_name": "LSTM-AE",
        "storage": "local_model_storage"
    },
    "anomaly_scoring": {
        "scores": "mahalanobis"  # mae_loss or mahalanobis
    },
    "mlflow_config": {
        "enabled": True,
        "experiment_name": "teaching-ADLM-v2",
        "tags": {"train_data": "normal"},
    },
    "inference": {
        "data_path": "data/verification/UNSW-NB15_1.csv",
        "ground_truth_cols": ['label'],  # or None for new data
        "model_path": "local_model_storage/lstmae",
        "transformer_path": "local_model_storage/transformer.sav"
    }
}
