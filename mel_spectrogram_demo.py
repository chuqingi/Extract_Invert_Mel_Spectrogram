import librosa
import soundfile
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


class MelExtractInvert:

    def __init__(self, audio, sr, n_fft, win_length, hop_length, n_mels, fmin, fmax, power):
        self.audio = audio
        self.sr = sr  # sample rate
        self.n_fft = n_fft  # fft points
        self.win_length = win_length  # frame length
        self.hop_length = hop_length  # frame shift
        self.n_mels = n_mels  # number of Mel banks
        self.fmin = fmin  # lowest frequency, remove low-frequency noise
        self.fmax = fmax  # highest frequency, sr / 2.0
        self.power = power  # exponent for the magnitude melspectrogram

    def extract_feature(self):
        spectrogram, phase = librosa.magphase(
            librosa.stft(self.audio, self.n_fft, self.hop_length, self.win_length))
        mel_basis = librosa.filters.mel(self.sr, self.n_fft, self.n_mels, fmin=self.fmin, fmax=self.fmax)
        spectrogram_power = spectrogram ** self.power
        mel_spectrogram = np.dot(mel_basis, spectrogram_power)
        return phase, mel_spectrogram

    def invert_feature(self, phase, mel_spectrogram):
        spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, self.sr, self.n_fft, self.power,
                                                          fmin=self.fmin, fmax=self.fmax)
        audio = librosa.istft(spectrogram * phase, self.hop_length, self.win_length)
        return audio


if __name__ == '__main__':
    audio, sr = librosa.load('./input.wav', sr=16000)

    MEL = MelExtractInvert(audio, sr, 320, 320, 160, 64, 100, 8000, 2)

    # Convert audio to mel spectrogram
    phase, mel_spectrogram = MEL.extract_feature()

    # Plot mel spectrogram
    mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure()
    librosa.display.specshow(mel_spec_db, sr=sr)
    plt.title('Mel spectrogram')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    '''
    processing mel_spectrogram
    '''

    # Convert mel spectrogram to audio
    reconstructed_audio = MEL.invert_feature(phase, mel_spectrogram)

    if len(audio) > len(reconstructed_audio):
        reconstructed_audio = np.resize(reconstructed_audio, audio.shape)
    else:
        reconstructed_audio = reconstructed_audio[:len(audio)]

    soundfile.write('./output.wav', reconstructed_audio, sr)
