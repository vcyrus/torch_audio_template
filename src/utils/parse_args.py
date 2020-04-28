from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "audio_path", type=str, help="path to dataset"
    ),
    parser.add_argument(
        "--win_length", 
        default=80, 
        type=int, 
        help="Spectrogram analysis window length (milliseconds)"
    ),
    parser.add_argument(
        "--hop_length", 
        default=20, 
        type=int, 
        help="Spectrogram analysis window hop length (milliseconds)"
    ),
    parser.add_argument(             
        "--n_bins", 
        default=128, 
        type=int, 
        help="Number of freq bins"
    ),
    parser.add_argument(
        "-fs", 
        default=22050, 
        type=int, 
        help="Audio resampling rate"
    ),
    parser.add_argument(
        "--audio_len", 
        default=1, 
        type=float, 
        help="Input audio length in seconds"
    ),
    parser.add_argument(
        "--batch_size", 
        default=1, 
        type=int, 
        help="training batch size"
    ),
    parser.add_argument(
        "--val_split",
        default=0.1,
        type=float,
        help="Fraction of data used for validation"
    ),
    parser.add_argument(
        "--test_split",
        default=0.1,
        type=float,
        help="Fraction of data used for testing"
    ),
    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
        help="Number of training epochs"
    ),
    parser.add_argument(
        "-lr",
        default=0.001, 
        type=float,
        help="Optimizer learning rate"
    ),
    parser.add_argument(
        "--use_cuda",
        default=True,
        type=bool,
        help="Use current CUDA GPU device if True, CPU if unspecified"
    ),
    parser.add_argument(
        "--out_path",
        default='models/',
        type=str,
        help="Path to save the trained model to"
    )
    parser.add_argument(
        "--dropout_rate",    
        default=0.3,
        type=float,
        help="Dropout rate applied to fully connected layers"
    )
    parser.add_argument(
        "--weight_decay",    
        default=0,
        type=float,
        help="Loss regularization weight decay"
    )
    parser.add_argument(
        "-patience",
        default=5,
        type=int,
        help="Early stopping patience"
    )
    parser.add_argument(
        "-transform",
        default='mel_spec',
        type=str,
        help="Audio transformation, mel_spec or cqt"
    )
    return parser.parse_args()  
