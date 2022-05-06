%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_251_chromophore_25 TDDFT with PBE1PBE functional

0 1
Mg   -2.512   34.320   26.710
C   -3.706   32.370   29.494
C   -1.096   36.452   28.856
C   -2.235   36.472   24.156
C   -4.454   32.242   24.657
N   -2.472   34.411   28.883
C   -2.842   33.436   29.825
C   -2.279   33.778   31.257
C   -1.516   35.114   31.007
C   -1.649   35.323   29.475
C   -1.968   36.325   31.783
C   -1.395   32.574   31.888
C   -1.813   32.123   33.293
C   -0.669   32.237   34.368
O   0.054   31.280   34.759
O   -0.640   33.491   34.830
N   -1.637   36.158   26.505
C   -1.110   36.895   27.488
C   -0.610   38.178   26.991
C   -1.042   38.204   25.620
C   -1.690   36.914   25.361
C   0.109   39.146   27.829
C   -0.905   39.451   24.691
O   -1.223   39.418   23.514
C   -0.355   40.758   25.166
N   -3.223   34.331   24.763
C   -2.855   35.305   23.841
C   -3.393   34.958   22.414
C   -3.787   33.440   22.551
C   -3.776   33.289   24.072
C   -4.458   35.913   21.783
C   -2.788   32.460   21.877
C   -3.333   31.256   21.151
N   -3.861   32.677   26.985
C   -4.559   31.955   26.059
C   -5.216   30.768   26.762
C   -4.866   30.866   28.087
C   -4.102   32.079   28.188
C   -5.976   29.695   26.016
C   -4.987   30.303   29.420
O   -5.621   29.260   29.684
C   -4.282   31.353   30.469
C   -5.239   32.034   31.425
O   -6.237   32.649   31.000
O   -4.967   31.732   32.760
C   -6.098   32.021   33.634
C   0.222   33.704   35.942
C   -0.282   34.928   36.657
C   0.051   35.427   37.854
C   1.023   34.675   38.818
C   -0.650   36.631   38.439
C   -1.770   36.397   39.492
C   -1.729   37.395   40.781
C   -1.454   36.600   42.108
C   -2.534   36.950   43.149
C   -0.008   36.811   42.638
C   0.502   35.633   43.537
C   0.883   35.946   45.025
C   2.176   35.252   45.423
C   2.180   34.820   46.907
C   3.349   36.121   44.950
C   4.641   35.371   44.762
C   5.110   35.624   43.304
C   5.387   34.303   42.566
C   6.581   34.522   41.686
C   4.121   33.908   41.727
H   -0.519   37.130   29.488
H   -2.044   37.203   23.368
H   -4.934   31.572   23.940
H   -3.190   33.893   31.844
H   -0.528   34.949   31.439
H   -2.834   36.131   32.416
H   -2.360   36.923   30.961
H   -1.189   36.745   32.420
H   -0.386   32.985   31.870
H   -1.367   31.747   31.179
H   -2.212   31.114   33.189
H   -2.623   32.762   33.647
H   -0.255   40.169   27.919
H   1.019   39.361   27.270
H   0.499   38.755   28.769
H   0.665   40.536   25.482
H   -0.917   41.023   26.062
H   -0.466   41.445   24.328
H   -2.506   35.025   21.784
H   -4.814   33.239   22.247
H   -4.509   36.750   22.479
H   -5.420   35.442   21.583
H   -4.086   36.406   20.884
H   -2.245   32.169   22.776
H   -2.065   32.930   21.210
H   -2.750   31.181   20.233
H   -4.377   31.369   20.857
H   -3.031   30.391   21.742
H   -5.976   29.976   24.963
H   -7.017   29.734   26.336
H   -5.547   28.719   26.242
H   -3.482   30.828   30.991
H   -6.594   31.149   34.058
H   -6.868   32.642   33.176
H   -5.697   32.605   34.462
H   1.253   33.954   35.690
H   0.150   32.926   36.702
H   -0.976   35.538   36.080
H   0.748   34.711   39.873
H   1.922   35.290   38.797
H   1.217   33.698   38.376
H   -1.070   37.182   37.597
H   0.064   37.316   38.896
H   -1.608   35.347   39.737
H   -2.763   36.548   39.068
H   -2.663   37.952   40.845
H   -1.082   38.270   40.724
H   -1.606   35.542   41.892
H   -2.143   37.655   43.883
H   -2.597   36.161   43.898
H   -3.499   37.257   42.747
H   0.040   37.778   43.139
H   0.717   36.807   41.824
H   1.364   35.225   43.010
H   -0.297   34.897   43.617
H   -0.018   35.578   45.515
H   0.966   37.031   45.095
H   2.269   34.324   44.859
H   2.955   35.289   47.513
H   2.268   33.735   46.953
H   1.285   35.075   47.476
H   3.574   36.775   45.792
H   3.002   36.703   44.096
H   4.533   34.289   44.838
H   5.460   35.735   45.382
H   5.977   36.264   43.467
H   4.435   36.225   42.694
H   5.649   33.580   43.340
H   6.732   33.779   40.903
H   7.446   34.513   42.351
H   6.506   35.506   41.225
H   3.936   32.964   42.239
H   4.224   33.820   40.645
H   3.228   34.454   42.032

