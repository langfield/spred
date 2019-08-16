from train.py import train, train_args
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser = train_args(parser)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
