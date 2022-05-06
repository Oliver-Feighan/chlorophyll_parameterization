%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_501_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.098   24.633   26.684
C   -3.702   26.367   29.522
C   -6.113   22.143   28.935
C   -6.480   23.096   24.133
C   -3.729   27.067   24.581
N   -4.871   24.330   29.015
C   -4.476   25.238   29.946
C   -4.870   24.819   31.353
C   -5.343   23.269   31.183
C   -5.428   23.221   29.630
C   -4.420   22.225   31.860
C   -5.972   25.743   31.973
C   -5.564   26.394   33.320
C   -6.482   25.916   34.587
O   -7.289   26.617   35.221
O   -6.011   24.680   35.008
N   -6.124   22.921   26.559
C   -6.454   21.955   27.569
C   -7.238   20.808   27.053
C   -7.372   21.152   25.668
C   -6.657   22.414   25.414
C   -7.704   19.615   27.890
C   -8.010   20.371   24.583
O   -8.306   20.800   23.473
C   -8.472   18.936   24.904
N   -5.236   25.127   24.715
C   -5.843   24.308   23.826
C   -5.574   24.690   22.397
C   -4.734   25.987   22.498
C   -4.583   26.133   24.029
C   -4.979   23.587   21.547
C   -5.554   27.258   21.863
C   -4.768   28.103   20.720
N   -3.719   26.233   26.928
C   -3.333   27.180   25.949
C   -2.554   28.225   26.623
C   -2.813   27.990   27.999
C   -3.404   26.732   28.126
C   -1.719   29.358   26.034
C   -2.392   28.463   29.348
O   -1.660   29.393   29.787
C   -3.081   27.468   30.360
C   -2.035   26.949   31.302
O   -0.946   26.579   30.965
O   -2.475   27.079   32.590
C   -1.384   26.911   33.583
C   -6.374   24.226   36.343
C   -7.414   23.062   36.258
C   -8.208   22.565   37.211
C   -8.073   22.838   38.650
C   -9.155   21.415   36.826
C   -10.633   21.906   36.573
C   -11.570   21.341   37.717
C   -12.970   21.084   37.103
C   -13.945   22.283   37.314
C   -13.551   19.827   37.804
C   -13.081   18.451   37.169
C   -12.248   17.597   38.170
C   -10.705   17.457   37.927
C   -10.011   18.026   39.187
C   -10.375   15.950   37.678
C   -8.891   15.706   37.398
C   -8.148   14.665   38.216
C   -6.622   14.648   37.801
C   -6.179   13.124   37.758
C   -5.693   15.376   38.778
H   -6.286   21.338   29.652
H   -6.914   22.633   23.244
H   -3.324   27.859   23.948
H   -3.949   24.825   31.936
H   -6.361   23.102   31.536
H   -3.941   21.589   31.115
H   -4.988   21.612   32.561
H   -3.647   22.627   32.515
H   -6.895   25.196   32.167
H   -6.221   26.572   31.310
H   -5.770   27.464   33.358
H   -4.505   26.326   33.567
H   -7.713   19.742   28.973
H   -7.200   18.671   27.683
H   -8.742   19.458   27.595
H   -7.620   18.317   25.185
H   -9.032   18.487   24.083
H   -9.156   18.989   25.751
H   -6.535   24.861   21.910
H   -3.751   25.858   22.046
H   -5.680   23.169   20.824
H   -4.667   22.698   22.095
H   -4.097   23.989   21.048
H   -5.869   27.995   22.601
H   -6.485   26.956   21.384
H   -4.001   27.429   20.338
H   -4.328   29.017   21.121
H   -5.402   28.282   19.852
H   -1.719   30.243   26.670
H   -2.230   29.702   25.134
H   -0.779   28.870   25.776
H   -3.744   28.083   30.968
H   -0.903   25.955   33.377
H   -1.918   26.795   34.526
H   -0.581   27.647   33.622
H   -6.805   25.045   36.919
H   -5.552   23.912   36.986
H   -7.603   22.722   35.239
H   -9.073   23.092   39.001
H   -7.374   23.572   39.053
H   -7.895   21.896   39.168
H   -9.125   20.653   37.605
H   -8.743   20.940   35.936
H   -10.933   21.508   35.604
H   -10.722   22.986   36.458
H   -11.703   22.058   38.526
H   -11.135   20.428   38.124
H   -12.965   20.936   36.023
H   -14.726   22.375   36.559
H   -13.333   23.172   37.463
H   -14.440   22.113   38.270
H   -14.630   19.912   37.675
H   -13.303   19.912   38.862
H   -12.534   18.600   36.238
H   -14.049   17.998   36.954
H   -12.658   16.593   38.054
H   -12.406   17.821   39.225
H   -10.514   18.078   37.052
H   -10.157   17.351   40.030
H   -10.455   18.947   39.563
H   -8.954   18.207   38.992
H   -10.858   15.612   36.761
H   -10.631   15.322   38.531
H   -8.372   16.656   37.528
H   -8.777   15.524   36.329
H   -8.696   13.812   37.816
H   -8.331   14.827   39.278
H   -6.427   15.052   36.807
H   -6.964   12.533   37.286
H   -6.107   12.809   38.799
H   -5.374   12.858   37.073
H   -5.180   16.196   38.275
H   -5.026   14.617   39.188
H   -6.372   15.759   39.539

