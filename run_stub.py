from coordinator import run_system
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract face crops from images using YOLO-based detection.")
    parser.add_argument("--targetter", action="store_true", help="Use the targetter to build (and test) a target profile from still images.")
    parser.add_argument("--build_test", action="store_true", help="Export to off-target face dataset instead of target set.")
    parser.add_argument("--no_nms", action="store_true", help="Export to off-target face dataset instead of target set.")
    args = parser.parse_args()

    if args.targetter:


        input_dir = config.INPUT_DIR
        output_dir = config.OUTPUT_DIR

        if args.build_test:
            input_dir = config.OFFTARGET_INPUT_DIR
            output_dir = config.OFFTARGET_OUTPUT_DIR
            packager = FacePackager(input_dir, output_dir, nms_mode=True)
        elif args.no_nms:
            packager = FacePackager(input_dir, output_dir, nms_mode=False)
        else:
            packager = FacePackager(input_dir, output_dir, nms_mode=True)

        packager.run()

    else:
        run_system()


if __name__ == "__main__":
    main()
