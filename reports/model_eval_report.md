# SiliconPilot AI — ML Model Training & Evaluation Report

Generated at: ""

## Iclr Layout Geometry Predictor
- **Dataset:** ICLR-ADAPT (Official 3,760 Observations)
- **Train / Test Split:** 2990 train / 770 test
### Metrics:
```json
{
  "per_target": {
    "Aeval": {
      "MAE": 3140.3041094766554,
      "RMSE": 7410.749250489379,
      "R2": -8.572400091069794,
      "NMAE": 0.2249575438323277,
      "unit": "\u00b5m\u00b2"
    },
    "Leq_signal": {
      "MAE": 2582.007820594942,
      "RMSE": 2813.5019952824623,
      "R2": -4.524225480482402,
      "NMAE": 0.3734554216342248,
      "unit": "\u00b5m"
    },
    "Leq_power": {
      "MAE": 7117.5939476383555,
      "RMSE": 12505.400657439855,
      "R2": -1.5311992124566705,
      "NMAE": 0.12617721204636723,
      "unit": "\u00b5m"
    },
    "Ncut_intermetal": {
      "MAE": 2556.1394773761244,
      "RMSE": 3877.1009303151163,
      "R2": -0.35189409642264935,
      "NMAE": 0.07927980514162039,
      "unit": "cuts"
    }
  },
  "overall": {
    "mean_R2": -3.7449297201078786,
    "mean_NMAE": 0.200967495663635,
    "mean_MAE": 3849.011338771519,
    "mean_RMSE": 6651.688208381703
  }
}
```

## Osiris Area Pex Predictor
- **Dataset:** OSIRIS Official Parquet (89,108 Variants)
- **Train / Test Split:** 71179 train / 17914 test
### Metrics:
```json
{
  "per_target": {
    "area": {
      "MAE": 251.610301446091,
      "RMSE": 872.5267282199922,
      "R2": 0.8216841895603765,
      "NMAE": 0.016886483400393348,
      "unit": "\u00b5m\u00b2"
    },
    "pex_score_low": {
      "MAE": 0.0015332143062293194,
      "RMSE": 0.0028792769853686956,
      "R2": 0.6126970535535597,
      "NMAE": 0.057758543500798704,
      "unit": "score"
    },
    "pex_score_mid": {
      "MAE": 0.0051881578378232774,
      "RMSE": 0.011738690585246646,
      "R2": 0.9136908714331616,
      "NMAE": 0.01849628383302906,
      "unit": "score"
    },
    "pex_score_high": {
      "MAE": 0.052003681473613214,
      "RMSE": 0.06971130731912964,
      "R2": 0.5036660135059061,
      "NMAE": 0.1028071128893686,
      "unit": "score"
    }
  },
  "overall": {
    "mean_R2": 0.712934532013251,
    "mean_NMAE": 0.04898710590589743,
    "mean_MAE": 62.91725662492717,
    "mean_RMSE": 218.1527643737205
  }
}
```

## Osiris Electrical Performance Predictor
- **Dataset:** OSIRIS Sizing & QoS Benchmark
- **Train / Test Split:** 285 train / 139 test
### Metrics:
```json
{
  "per_target": {
    "dc_gain": {
      "MAE": 2.0580950239808113,
      "RMSE": 2.938417324981942,
      "R2": 0.7942252983110926,
      "NMAE": 0.06488790246394953,
      "unit": "dB"
    },
    "gain_bandwidth_product": {
      "MAE": 4104925.225185252,
      "RMSE": 6526847.88423563,
      "R2": 0.7816052302558458,
      "NMAE": 0.04521335581605323,
      "unit": "Hz"
    },
    "phase_margin": {
      "MAE": 4.854429196642692,
      "RMSE": 5.782588675130772,
      "R2": 0.4502954771997243,
      "NMAE": 0.1494415430658572,
      "unit": "\u00b0"
    },
    "slew_rate": {
      "MAE": 2.6165723081534766,
      "RMSE": 3.748025760742661,
      "R2": 0.6781933675244707,
      "NMAE": 0.05920623765456726,
      "unit": "V/\u00b5s"
    },
    "cmrr": {
      "MAE": 3.441807853717028,
      "RMSE": 4.29499534157813,
      "R2": 0.20029732319385019,
      "NMAE": 0.14233344996824113,
      "unit": "dB"
    }
  },
  "overall": {
    "mean_R2": 0.5809233392969968,
    "mean_NMAE": 0.09221649779373367,
    "mean_MAE": 820987.6392179269,
    "mean_RMSE": 1305372.9296525463
  }
}
```

## Layout Verification Classifier
- **Dataset:** OSIRIS Layout Verification
- **Train / Test Split:** 336 train / 164 test
### Metrics:
```json
{
  "Accuracy": 0.8475609756097561,
  "Precision": 0.8475609756097561,
  "Recall": 1.0,
  "F1": 0.9174917491749175
}
```

