import argparse
import os
from converters import (
    QAConverter, 
    DialogueConverter, 
    TCMConverter, 
    TranslationConverter, 
    QwenConverter
)

def main():
    parser = argparse.ArgumentParser(description="Medical Data SFT Converter System")
    parser.add_argument('input_path', type=str, help='Path to the raw data file')
    parser.add_argument('mode', type=str, help='Conversion mode', 
                        choices=[
                            'multiquestion', 
                            'inquiry', 'inquiry2', 'raw-dialogue', 
                            'common-multi-dialogue',
                            'TCM-inquiry',
                            'translation', 'translation2',
                            'qwen-llm'
                        ])
    
    args = parser.parse_args()
    
    # Determine Output Path
    filename = os.path.basename(args.input_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join('sft', 'output_sft_sample', f"Converted-{name}.jsonl")
    
    print(f"=== Starting Conversion ===")
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")

    # Dispatcher
    if args.mode == 'multiquestion':
        converter = QAConverter(args.input_path, output_path)
    
    elif args.mode in ['inquiry', 'inquiry2', 'raw-dialogue', 'common-multi-dialogue']:
        converter = DialogueConverter(args.input_path, output_path, mode=args.mode)
        
    elif args.mode == 'TCM-inquiry':
        converter = TCMConverter(args.input_path, output_path)
        
    elif args.mode in ['translation', 'translation2']:
        converter = TranslationConverter(args.input_path, output_path, mode=args.mode)
        
    elif args.mode == 'qwen-llm':
        converter = QwenConverter(args.input_path, output_path)
    
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # Execute
    converter.process()
    print("=== Done ===")

if __name__ == "__main__":
    main()