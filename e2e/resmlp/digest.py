"""
Simply reads the TVM test results and prints a digest
"""
import pandas as pd


def main():
    numerical = pd.read_csv("./numerical.csv")
    pred = pd.read_csv("./pred.csv")

    use_accel = (numerical["accel_time"][0] != "None")

    print(f"Average Relay time: {numerical['relay_time'].mean()}")
    print(f"Average PT time: {numerical['pt_time'].mean()}")
    if use_accel:
        print(f"Average accelerated time: {numerical['accel_time'].mean()}")

    # cast to float so everything else becomes a float
    total = float(len(pred["relay_faithful"]))
    pt_correct = pred[pred.pt_correct == True].shape[0]
    relay_correct = pred[pred.relay_correct == True].shape[0]
    relay_faithful = pred[pred.relay_faithful == True].shape[0]
    print(f"PT accuracy: {(pt_correct/total)*100}")
    print(f"Relay accuracy: {(relay_correct/total)*100}%")
    print(f"Relay faithfulness: {(relay_faithful/total) * 100}%")
    if use_accel:
        accel_faithful = pred[pred.accel_faithful == True].shape[0]
        accel_correct = pred[pred.accel_correct == True].shape[0]
        print(f"Accelerator faithfulness: {(accel_faithful/total)*100}%")
        print(f"Accelerator accuracy: {(accel_correct/total)*100}%")


if __name__ == "__main__":
    main()
