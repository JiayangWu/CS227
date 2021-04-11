import py_ts_data
import os
import shutil

valid_datasets = ['Haptics', 'SyntheticControl', 'Worms', 'Computers', 'HouseTwenty', 'Chinatown', 'UWaveGestureLibraryAll', 'Strawberry', 'Car', 'GunPointAgeSpan', 'BeetleFly', 'Wafer', 'CBF', 'Adiac', 'ItalyPowerDemand', 'Yoga', 'Trace', 'PigAirwayPressure', 'ShapesAll', 'Beef', 'Mallat', 'GunPointOldVersusYoung', 'MiddlePhalanxTW', 'Meat', 'Herring', 'MiddlePhalanxOutlineCorrect', 'InsectEPGRegularTrain', 'FordA', 'SwedishLeaf', 'InlineSkate', 'UMD', 'CricketY', 'WormsTwoClass', 'SmoothSubspace', 'OSULeaf', 'Ham', 'CricketX', 'SonyAIBORobotSurface1', 'ToeSegmentation1', 'ScreenType', 'PigArtPressure', 'SmallKitchenAppliances', 'Crop', 'MoteStrain', 'MelbournePedestrian', 'ECGFiveDays', 'Wine', 'SemgHandMovementCh2', 'FreezerSmallTrain', 'UWaveGestureLibraryZ', 'NonInvasiveFetalECGThorax1', 'TwoLeadECG', 'Lightning7', 'Phoneme', 'SemgHandSubjectCh2', 'MixedShapes', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'FacesUCR', 'ECG5000', 'HandOutlines', 'GunPointMaleVersusFemale', 'Coffee', 'Rock', 'MixedShapesSmallTrain', 'FordB', 'FiftyWords', 'InsectWingbeatSound', 'MedicalImages', 'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'ChlorineConcentration', 'Plane', 'ACSF1', 'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGSmallTrain', 'EOGVerticalSignal', 'CricketZ', 'FaceFour', 'RefrigerationDevices', 'GunPoint', 'ECG200', 'ToeSegmentation2', 'WordSynonyms', 'Fungi', 'BirdChicken', 'SemgHandGenderCh2', 'OliveOil', 'BME', 'LargeKitchenAppliances', 'SonyAIBORobotSurface2', 'Lightning2', 'EthanolLevel', 'UWaveGestureLibraryX', 'FreezerRegularTrain', 'Fish', 'ProximalPhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'UWaveGestureLibraryY', 'FaceAll', 'StarlightCurves', 'ElectricDevices', 'Earthquakes', 'PowerCons', 'DiatomSizeReduction', 'CinCECGtorso', 'PigCVP', 'ProximalPhalanxTW']

for item in os.walk("data"):
    
    if "/" in item[0] and item[0].split("/")[1] not in valid_datasets:
        #     print(item, item[0])
            shutil.rmtree(item[0])


all_datasets =  ['Haptics', 'SyntheticControl', 'Worms', 'Computers', 'HouseTwenty', 'GestureMidAirD3', 'LSST', 'SelfRegulationSCP2', 'Chinatown', 'UWaveGestureLibraryAll', 'Strawberry', 'Car', 'GunPointAgeSpan', 'FaceDetection', 'GestureMidAirD2', 'BeetleFly', 'MotorImagery', 'Wafer', 'CBF', 'Adiac', 'PenDigits', 'ItalyPowerDemand', 'Yoga', 'AllGestureWiimoteY', 'Trace', 'PigAirwayPressure', 'ShapesAll', 'Beef', 'GesturePebbleZ2', 'Mallat', 'GunPointOldVersusYoung', 'MiddlePhalanxTW', 'AllGestureWiimoteX', 'Meat', 'Libras', 'Herring', 'MiddlePhalanxOutlineCorrect', 'InsectEPGRegularTrain', 'FordA', 'SwedishLeaf', 'InlineSkate', 'UMD', 'CricketY', 'InsectWingbeat', 'WormsTwoClass', 'Cricket', 'SmoothSubspace', 'OSULeaf', 'Ham', 'CricketX', 'SonyAIBORobotSurface1', 'ToeSegmentation1', 'Handwriting', 'ScreenType', 'PigArtPressure', 'SmallKitchenAppliances', 'Crop', 'MoteStrain', 'ArticularyWordRecognition', 'StandWalkJump', 'MelbournePedestrian', 'CharacterTrajectories', 'ECGFiveDays', 'Wine', 'SemgHandMovementCh2', 'FreezerSmallTrain', 'UWaveGestureLibraryZ', 'Ering', 'NonInvasiveFetalECGThorax1', 'TwoLeadECG', 'Lightning7', 'HandMovementDirection', 'Phoneme', 'SelfRegulationSCP1', 'SemgHandSubjectCh2', 'JapaneseVowels', 'Heartbeat', 'RacketSports', 'MixedShapes', 'MiddlePhalanxOutlineAgeGroup', 'GestureMidAirD1', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'FacesUCR', 'EigenWorms', 'ECG5000', 'ShakeGestureWiimoteZ', 'FingerMovements', 'GesturePebbleZ1', 'HandOutlines', 'GunPointMaleVersusFemale', 'PEMS-SF', 'Epilepsy', 'Coffee', 'Rock', 'MixedShapesSmallTrain', 'AllGestureWiimoteZ', 'FordB', 'FiftyWords', 'NATOPS', 'InsectWingbeatSound', 'AtrialFibrillation', 'MedicalImages', 'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup', 'EOGHorizontalSignal', 'TwoPatterns', 'ChlorineConcentration', 'Plane', 'ACSF1', 'SpokenArabicDigits', 'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup', 'InsectEPGSmallTrain', 'PickupGestureWiimoteZ', 'EOGVerticalSignal', 'CricketZ', 'FaceFour', 'RefrigerationDevices', 'PLAID', 'GunPoint', 'ECG200', 'ToeSegmentation2', 'WordSynonyms', 'Fungi', 'BirdChicken', 'EthanolConcentration', 'SemgHandGenderCh2', 'OliveOil', 'BME', 'BasicMotions', 'LargeKitchenAppliances', 'SonyAIBORobotSurface2', 'Lightning2', 'EthanolLevel', 'UWaveGestureLibrary', 'UWaveGestureLibraryX', 'FreezerRegularTrain', 'Fish', 'ProximalPhalanxOutlineCorrect', 'NonInvasiveFetalECGThorax2', 'UWaveGestureLibraryY', 'FaceAll', 'StarlightCurves', 'ElectricDevices', 'Earthquakes', 'PowerCons', 'DiatomSizeReduction', 'CinCECGtorso', 'PigCVP', 'ProximalPhalanxTW']

# valid_datasets = []
# small_datasets = []
# for dataset in sorted(valid_datasets):
# #     if py_ts_data.data_info(dataset)['n_timestamps'] > 0 and \
# #         py_ts_data.data_info(dataset)['n_variables'] == 1:
# #         # valid_datasets.append(dataset)
# #         # print(py_ts_data.data_info(dataset))
# #         print(dataset, py_ts_data.data_info(dataset)['train_size'])
#         X_train, y_train, X_test, y_test, info = py_ts_data.load_data(dataset, variables_as_channels=True)
#         x, y = X_train.shape[0], X_train.shape[1]
#         if x * y < 10000:
#                 small_datasets.append(dataset)

# print(small_datasets)
                # print("Dataset {} shape: Train: {}, Test: {}".format(dataset, X_train.shape, X_test.shape))





