import argparse

from tlm.generate import generate
from tlm.model import TransformerLM
from tlm.tokenizer import Tokenizer
from tlm.train.checkpoint import load_checkpoint, load_config
from tlm.train.train import pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="workspace/tinystories_bpe_vocab.pkl")
    parser.add_argument("--merges", default="workspace/tinystories_bpe_merges.pkl")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()

    device = pick_device()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])

    config = load_config(args.checkpoint)
    model = TransformerLM(config).to(device)
    load_checkpoint(args.checkpoint, model)

    eos_token_id = tokenizer.vocab_bytes_to_id[b"<|endoftext|>"]
    prompt_ids = tokenizer.encode(args.prompt)

    generated_ids = list(prompt_ids)
    printed_text = tokenizer.decode(generated_ids)
    print(printed_text, end="", flush=True)

    def on_token(token_id: int) -> None:
        nonlocal printed_text
        generated_ids.append(token_id)
        if token_id == eos_token_id:
            return
        # redecode the whole sequence each time so a token that's a partial
        # multi-byte UTF-8 sequence resolves correctly once enough bytes arrive
        text_so_far = tokenizer.decode(generated_ids)
        print(text_so_far[len(printed_text) :], end="", flush=True)
        printed_text = text_so_far

    generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        on_token=on_token,
    )
    print()


if __name__ == "__main__":
    main()
