import cv2
import numpy as np
import mido

class VirtualKeyboard:
    draw_width      : int   = 512
    draw_height     : int   = 256
    last_x          : float = 0.5
    last_y          : float = 0.5
    midi_out_idx    : int   = 0
    midi_port               = None
    keyboard_frame  : np.array

    def init(self):
        self.keyboard_frame = np.empty([self.draw_height, self.draw_width], dtype = np.uint8)

        def on_changed(value):
            pass

        cv2.namedWindow('Keyboard')
        cv2.createTrackbar("Num Octaves", "Keyboard", 1, 4, on_changed)
        cv2.setTrackbarMin("Num Octaves", "Keyboard", 1)

        cv2.createTrackbar("MIDI Output", "Keyboard", 0, 0, on_changed)

    def shutdown(self):
        self.__close_current_midi_port()

    def put_position(self, x : float, y : float):
        #print(f"put {x},{y}")
        self.last_x = 1.0 - np.clip(x, 0.0, 1.0) # flip X
        self.last_y =       np.clip(y, 0.0, 1.0)

    def update_and_draw(self):
        keyboard_image = self.keyboard_frame
        keyboard_image.fill(255)

        font            = cv2.FONT_HERSHEY_SIMPLEX
        font_size       = 0.75
        font_color      = (0, 0, 0)
        font_thickness  = 2

        midi_outputs = ["None"] + mido.get_output_names()
        cv2.setTrackbarMax("MIDI Output", "Keyboard", len(midi_outputs) - 1)
        midi_out_idx = cv2.getTrackbarPos("MIDI Output", "Keyboard")
        cv2.putText(keyboard_image, midi_outputs[midi_out_idx], (10, 30), font, font_size, font_color, font_thickness, cv2.LINE_AA)

        if midi_out_idx != self.midi_out_idx:
            if self.midi_port != None:
                self.__close_current_midi_port()
            if midi_out_idx > 0:
                self.midi_port = mido.open_output(midi_outputs[midi_out_idx])
            self.midi_out_idx = midi_out_idx

        keys_count = cv2.getTrackbarPos("Num Octaves", "Keyboard") * 7
        key_width = self.draw_width / keys_count

        white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

        note_to_play = -1
        for i in range(keys_count):
            key_pos = int(key_width * i)
            cv2.line(keyboard_image, (key_pos, 0), (key_pos, self.draw_height), 0)

            key_note = i % len(white_keys)
            cv2.putText(keyboard_image, white_keys[key_note], (key_pos + 10, self.draw_height - 10), font, font_size, font_color, font_thickness, cv2.LINE_AA)

            eye_x = self.last_x * self.draw_width
            if eye_x > key_pos and eye_x < (key_pos + key_width):
                note_to_play = key_note

        self.__all_notes_off()
        if self.midi_port != None and note_to_play >= 0:
            octave_idx      = note_to_play // 7
            note_in_octave  = note_to_play % 7

            white_offset    = [0, 1, 2, 2, 3, 4, 5]
            white_note      = note_in_octave + white_offset[note_in_octave]

            note = 60 + (octave_idx * 12) + white_note
            print("Note:", note)
            self.midi_port.send(mido.Message('note_on', note=note))

        pos_x = int(self.draw_width  * self.last_x)
        pos_y = int(self.draw_height * self.last_y)

        cv2.circle(keyboard_image, (pos_x, pos_y), 4, 0, 6)
        cv2.imshow("Keyboard", self.keyboard_frame)


    def __all_notes_off(self):
        if self.midi_port != None:
            for i in range(12*8):
                self.midi_port.send(mido.Message('note_off', note=i))

    def __close_current_midi_port(self):
        if self.midi_port != None:
            self.__all_notes_off()
            self.midi_port.close()
            self.midi_port = None