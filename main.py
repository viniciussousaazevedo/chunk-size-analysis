import scripts.ct_gen_chunk_size_analysis as ct_gen
import scripts.qa_chunk_size_analysis as qa
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("-qa", action="store_true", help="Evaluate Question Answering")
    parser.add_argument("-tc", action="store_true", help="Evaluate Test Case Generation")
    
    args = parser.parse_args()

    if args.qa:
        qa.run()
    elif args.tc:
        ct_gen.run()
    else:
        print("Please specify either -qa for Question Answering or -tc for Test Case Generation.")