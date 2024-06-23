import sys
import ffnn
import rnn

def main():
    if len(sys.argv) != 2:
        print("Usage: python pos_tagger.py [-f | -r]")
        return
    
    tagger_type = sys.argv[1]
    
    if tagger_type == "-f":
        ffnn.main()
    elif tagger_type == "-r":
        rnn.main()
    else:
        print("Invalid argument. Use -f for FFNN or -r for RNN.")
        return
    

if __name__ == "__main__":
    main()

