%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_101_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   15.777   51.451   25.602
C   17.236   50.316   28.600
C   12.936   52.197   27.385
C   14.270   52.010   22.735
C   18.544   50.105   23.903
N   15.149   51.181   27.757
C   15.993   50.911   28.823
C   15.431   51.440   30.182
C   13.951   51.754   29.810
C   14.015   51.789   28.240
C   12.695   50.890   30.256
C   16.268   52.698   30.604
C   16.073   53.249   31.955
C   14.885   52.798   32.864
O   14.731   51.674   33.242
O   13.970   53.834   33.124
N   13.871   51.966   25.125
C   12.850   52.298   26.016
C   11.651   52.593   25.269
C   11.950   52.372   23.892
C   13.429   52.105   23.847
C   10.346   52.934   25.913
C   11.074   52.396   22.625
O   11.478   52.047   21.549
C   9.647   52.787   22.757
N   16.331   51.061   23.540
C   15.582   51.546   22.519
C   16.238   51.370   21.164
C   17.548   50.574   21.597
C   17.533   50.585   23.116
C   15.280   50.729   20.118
C   18.848   51.127   20.970
C   19.345   50.247   19.836
N   17.549   50.343   26.112
C   18.594   49.899   25.323
C   19.624   49.343   26.163
C   19.144   49.411   27.442
C   17.856   50.030   27.409
C   20.916   48.905   25.635
C   19.446   49.207   28.805
O   20.507   48.868   29.368
C   18.246   49.826   29.556
C   17.554   48.996   30.625
O   16.748   48.100   30.509
O   18.114   49.309   31.848
C   17.837   48.419   32.962
C   12.680   53.333   33.726
C   11.859   54.552   34.066
C   11.945   55.196   35.271
C   13.054   55.061   36.274
C   10.850   56.156   35.662
C   9.566   55.468   36.009
C   8.363   56.260   35.454
C   7.747   57.238   36.506
C   6.740   56.520   37.467
C   7.154   58.540   35.920
C   7.652   59.819   36.645
C   9.013   60.227   36.126
C   9.275   61.697   36.247
C   8.536   62.277   35.032
C   10.811   61.935   36.236
C   11.276   62.822   37.297
C   11.400   64.270   36.768
C   10.842   65.268   37.776
C   9.279   65.460   37.504
C   11.634   66.580   37.686
H   12.079   52.416   28.026
H   13.825   52.211   21.759
H   19.414   49.781   23.327
H   15.406   50.642   30.925
H   13.837   52.811   30.049
H   12.329   50.411   29.349
H   11.928   51.476   30.762
H   13.065   50.123   30.936
H   16.322   53.433   29.801
H   17.310   52.377   30.591
H   15.919   54.297   31.698
H   17.030   53.067   32.443
H   9.985   53.820   25.391
H   10.553   53.313   26.914
H   9.590   52.150   25.894
H   9.614   53.859   22.951
H   9.189   52.125   23.492
H   9.213   52.583   21.778
H   16.510   52.334   20.735
H   17.404   49.576   21.184
H   14.421   50.319   20.649
H   15.762   49.901   19.599
H   14.816   51.548   19.568
H   19.699   51.299   21.629
H   18.548   52.073   20.520
H   20.130   49.618   20.255
H   19.902   50.862   19.128
H   18.600   49.764   19.203
H   21.332   49.845   25.272
H   20.886   48.128   24.872
H   21.587   48.435   26.355
H   18.614   50.740   30.022
H   17.210   48.950   33.679
H   18.738   48.101   33.488
H   17.300   47.490   32.771
H   12.857   52.669   34.573
H   12.104   52.783   32.982
H   10.934   54.691   33.507
H   13.692   54.227   35.982
H   12.407   54.666   37.057
H   13.637   55.916   36.617
H   10.772   56.928   34.897
H   11.181   56.765   36.503
H   9.378   55.328   37.074
H   9.434   54.514   35.498
H   7.552   55.626   35.096
H   8.694   56.909   34.643
H   8.469   57.495   37.281
H   6.631   57.105   38.380
H   7.040   55.506   37.733
H   5.740   56.551   37.036
H   6.073   58.585   36.053
H   7.426   58.707   34.877
H   7.808   59.534   37.685
H   6.860   60.568   36.628
H   9.247   59.714   35.193
H   9.668   59.797   36.883
H   8.697   62.063   37.096
H   7.666   62.883   35.283
H   8.095   61.522   34.381
H   9.137   62.948   34.419
H   11.185   62.308   35.283
H   11.359   61.020   36.466
H   12.252   62.429   37.579
H   10.520   62.738   38.078
H   10.921   64.424   35.801
H   12.474   64.387   36.618
H   10.960   64.941   38.810
H   8.812   65.143   38.437
H   9.023   64.923   36.591
H   9.074   66.507   37.278
H   12.633   66.499   37.259
H   11.718   67.070   38.656
H   11.281   67.358   37.009

