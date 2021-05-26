clear all;

folder = '..\..\_Release';
packingSerializer = PackingSerializer();

% Read the config
configFilePath = fullfile(folder, 'generation.conf');
config = packingSerializer.ReadConfig(configFilePath);

% Reading the packing
packingFilePath = fullfile(folder, 'packing_scaled_by_script_2.xyzd');
% packingFilePath = fullfile(folder, 'packing_init.xyzd');
[packing, floatDataType] = packingSerializer.ReadPacking(packingFilePath, config.ParticlesCount);

% Reading packing.nfo
infoFilePath = fullfile(folder, 'packing.nfo');
packingInfo = packingSerializer.ReadPackingInfo(infoFilePath);

% Save coordinates and diameters to CSV files
coordinates = packing.ParticleCoordinates;
writematrix(coordinates, "coordinates3.csv")
diameters = packing.ParticleDiameters;
writematrix(diameters, "diameters3.csv")