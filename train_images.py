import os
import argparse
from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model


def build_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_mobilenetv2_classifier(input_shape: Tuple[int, int, int], num_classes: int, weights: str = 'imagenet', alpha: float = 1.0):
    base: Model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        alpha=alpha
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN image classifier from a directory (ImageFolder)")
    parser.add_argument('--train_dir', type=str, required=True, help='Training directory; subfolders are class names')
    parser.add_argument('--val_dir', type=str, required=False, default=None, help='Validation directory; if empty, will split from training set')
    parser.add_argument('--img_size', type=int, nargs=2, default=[32, 32], help='Input size (w h)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split when --val_dir is not provided')
    parser.add_argument('--output', type=str, default=os.path.join('artifacts_cnn'), help='Output directory')
    parser.add_argument('--model_name', type=str, default='cnn', help='Model name prefix')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--price_json', type=str, default='person_info.json', help='Optional JSON mapping class name to price')
    parser.add_argument('--also_h5', action='store_true', help='Additionally save models in legacy HDF5 (.h5) format')
    parser.add_argument('--backbone', type=str, default='simple', choices=['simple', 'mobilenetv2'], help='Model backbone')
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Epochs to train with backbone frozen (mobilenetv2)')
    parser.add_argument('--finetune', action='store_true', help='Unfreeze backbone for fine-tuning after freeze_epochs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    img_width, img_height = args.img_size
    input_shape = (img_height, img_width, 3)

    if args.augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=(args.val_split if args.val_dir is None else 0.0),
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=(args.val_split if args.val_dir is None else 0.0),
        )

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    if args.val_dir is None:
        train_gen = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(img_height, img_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
        )
        val_gen = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(img_height, img_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(img_height, img_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=True,
        )
        val_gen = val_datagen.flow_from_directory(
            args.val_dir,
            target_size=(img_height, img_width),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=False,
        )

    num_classes = len(train_gen.class_indices)
    if args.backbone == 'mobilenetv2':
        # If user forgot to set 224, still build with provided size
        if img_height < 96 or img_width < 96:
            print('Warning: MobileNetV2 usually expects >=96, recommended 224x224')
        model, base = build_mobilenetv2_classifier(input_shape=input_shape, num_classes=num_classes)
        # Freeze backbone first
        base.trainable = False
    else:
        model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)

    best_path = os.path.join(args.output, f'{args.model_name}_best.keras')
    final_path = os.path.join(args.output, f'{args.model_name}_final.keras')
    history_csv = os.path.join(args.output, f'{args.model_name}_history.csv')
    classes_txt = os.path.join(args.output, f'{args.model_name}_classes.txt')
    classes_with_price_txt = os.path.join(args.output, f'{args.model_name}_classes_with_price.txt')
    best_h5 = os.path.join(args.output, f'{args.model_name}_best.h5')
    final_h5 = os.path.join(args.output, f'{args.model_name}_final.h5')

    # Save class mapping (index -> name)
    with open(classes_txt, 'w', encoding='utf-8') as f:
        for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{name}\n")

    # Load prices and print a table of class -> price
    prices = {}
    if args.price_json and os.path.exists(args.price_json):
        try:
            with open(args.price_json, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                # Expect format: {"ClassName": {"Giá": "..."}, ...}
                for cname, info in data.items():
                    if isinstance(info, dict) and ('Giá' in info or 'Price' in info):
                        prices[cname] = info.get('Giá', info.get('Price', 'N/A'))
        except Exception:
            prices = {}

    if prices:
        lines = []
        print("Classes and prices:")
        for name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
            price = prices.get(name, 'N/A')
            line = f"{idx}\t{name}\t{price}"
            lines.append(line)
            print(line)
        with open(classes_with_price_txt, 'w', encoding='utf-8') as f:
            f.write("index\tclass\tprice\n")
            for line in lines:
                f.write(line + "\n")

    callbacks = [
        ModelCheckpoint(best_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True, verbose=1),
        CSVLogger(history_csv)
    ]

    histories = []
    total_epochs = args.epochs
    if args.backbone == 'mobilenetv2' and args.freeze_epochs > 0:
        e1 = min(args.freeze_epochs, total_epochs)
        print(f'Training with frozen backbone for {e1} epochs...')
        h1 = model.fit(train_gen, validation_data=val_gen, epochs=e1, callbacks=callbacks, verbose=1)
        histories.append(h1)
        total_epochs -= e1
        if args.finetune and total_epochs > 0:
            print('Unfreezing backbone for fine-tuning...')
            # Unfreeze backbone
            try:
                base.trainable = True
            except Exception:
                for layer in model.layers:
                    layer.trainable = True
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            h2 = model.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=callbacks, verbose=1)
            histories.append(h2)
        history = histories[-1]
    else:
        history = model.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=callbacks, verbose=1)

    model.save(final_path)
    print(f"Saved model: {final_path} (best checkpoint: {best_path})")

    # Optionally export .h5 models
    if args.also_h5:
        try:
            from tensorflow.keras.models import load_model as _load_model
            # save best as .h5
            if os.path.exists(best_path):
                _best_model = _load_model(best_path)
                _best_model.save(best_h5, save_format='h5')
                print(f"Saved HDF5 best model: {best_h5}")
            # save final as .h5
            model.save(final_h5, save_format='h5')
            print(f"Saved HDF5 final model: {final_h5}")
        except Exception as e:
            print(f"Failed to save .h5 models: {e}")

    # Plot and save training curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history.get('accuracy', []), label='train_acc')
        axes[0].plot(history.history.get('val_accuracy', []), label='val_acc')
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Acc')
        axes[0].legend()

        axes[1].plot(history.history.get('loss', []), label='train_loss')
        axes[1].plot(history.history.get('val_loss', []), label='val_loss')
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(args.output, f'{args.model_name}_training_curves.png')
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"Saved training curves: {plot_path}")
    except Exception as e:
        print(f"Failed to plot training curves: {e}")


if __name__ == '__main__':
    main()


