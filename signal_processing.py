import librosa

signal, sr = librosa.load(path, sr = sample_rate, mono = mono)

# trimming head and tail
signal, _ = librosa.effects.trim(signal, top_db = 60, frame_length = frame_length, hop_length = hop_length)

filter_banks = librosa.feature.melspectrogram(signal, sr = sample_rate, n_fft = NFFT, 
                                              hop_length = hop_length, n_mels = nfilt, fmax = 8000, center = False)

filter_banks = librosa.power_to_db(filter_banks, ref = np.max)

filter_banks -= np.mean(filter_banks, -1, keepdims = True)
filter_banks /= (np.std(filter_banks, -1, keepdims = True) + 1E-8)

filter_banks = filter_banks.T


