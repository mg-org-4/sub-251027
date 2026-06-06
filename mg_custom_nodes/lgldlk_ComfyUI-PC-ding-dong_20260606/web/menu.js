import { app } from '../../../scripts/app.js';

import { api } from '../../scripts/api.js';
import { $el } from '../../scripts/ui.js';
import { fetchAndPlayAudioSingle, get_video_files, set_play_type, fetchAndPlayAudio, play_ding_dong_text, request } from './utils.js';

const id_prefix = '⏰Ding_Dong';
const id_music_prefix = `${id_prefix}.music`;
let selectedAudio = null;
let volume = 100;
let open = true;
// all | one
let play_type = 'all';
let fail_tip = false;
const select_menu_name = `${id_music_prefix}.name`;
const select_menu_name_options = $el('select', {
  onchange: (e) => {
    const value = e.target.value;
    if (value === selectedAudio) {
      return;
    }
    set_select_menu_name_value(value);
  },
});

function set_select_menu_name_options(options) {
  select_menu_name_options.innerHTML = '';
  options.forEach((item) => {
    select_menu_name_options.appendChild($el('option', { value: item.value, textContent: item.text, selected: item.selected }));
  });
}
function set_select_menu_name_value(value) {
  selectedAudio = value;
  select_menu_name_options.value = value;
  app.ui.settings.setSettingValue(select_menu_name, value);
}

function get_video_files_list() {
  return get_video_files().then((res) =>
    res
      .map((item) => ({
        value: item,
        text: item,
        selected: selectedAudio === item,
      }))
      .concat({
        value: null,
        text: 'None',
        selected: selectedAudio === null,
      })
  );
}

const fileInput = $el('input', {
  type: 'file',
  accept: '.mp4,.avi,.mov,.mkv,.webm,.mp3',
  style: { display: 'none' },
  parent: document.body,
  onchange: async () => {
    const file = fileInput.files[0];
    const menu_name = `${id_music_prefix}.name`;
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav'];
    if (validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      const reader = new FileReader();
      reader.onload = async () => {};
      reader.readAsArrayBuffer(file);
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await api.fetchApi('/pc_upload_video', {
          method: 'POST',
          body: formData,
        });

        const result = await response.json();
        if (result.success) {
          set_select_menu_name_options(await get_video_files_list());
        } else {
          console.error('Upload failed:', result.error);
        }
      } catch (err) {
        console.error('Error uploading file:', err);
      }
    }
  },
});
app.registerExtension({
  name: id_prefix + '.menu',
  init() {
    get_video_files_list().then(async (res) => {
      try {
        set_select_menu_name_options(res);
        app.ui.settings.addSetting({
          id: `${id_music_prefix}.name`,
          name: 'play music name',
          tooltip: 'select music name',
          defaultValue: res[0].value,
          type: () => select_menu_name_options,
          onChange(value) {
            if (value === selectedAudio) {
              return;
            }
            set_select_menu_name_value(value);
          },
        });
        app.ui.settings.addSetting({
          id: `${id_music_prefix}.play`,
          name: 'play music select',
          type: () => {
            return $el('button', {
              textContent: 'play',
              onclick: () => {
                fetchAndPlayAudioSingle(selectedAudio, volume / 100);
              },
            });
          },
        });
      } catch (e) {
        console.error('get_video_files error', e);
      }
    });
    api.addEventListener('pc.play_ding_dong_audio', ({ detail }) => {
      if (detail.status === 'error' && !fail_tip) {
        return;
      }
      if (selectedAudio && open) {
        fetchAndPlayAudioSingle(selectedAudio, volume / 100);
      }
    });
    api.addEventListener('pc.play_ding_dong_mui', ({ detail }) => {
      fetchAndPlayAudio(detail.music, detail.volume / 100);
    });
    api.addEventListener('pc.play_ding_dong_text', ({ detail }) => {
      play_ding_dong_text(detail.text, detail.pitch, detail.rate, detail.volume);
    });

    app.ui.settings.addSetting({
      id: `${id_music_prefix}.volume`,
      name: 'Volume',
      type: 'slider',
      attrs: {
        min: 0,
        max: 100,
        step: 1,
      },
      tooltip: 'set ding dong volume',
      defaultValue: 100,
      onChange(v) {
        volume = v;
      },
    });

    app.ui.settings.addSetting({
      id: `${id_music_prefix}.open`,
      name: 'open',
      type: 'boolean',
      defaultValue: true,
      onChange(v) {
        open = v;
      },
    });

    app.ui.settings.addSetting({
      id: `${id_music_prefix}.play_type`,
      name: 'play type',
      tooltip: 'play after all workflows finish or after each single workflow finishes',
      type: 'combo',
      defaultValue: 'all',
      options: [
        { value: 'all', text: 'all' },
        { value: 'one', text: 'one' },
      ],
      onChange(v) {
        set_play_type(v);
        play_type = v;
      },
    });
    app.ui.settings.addSetting({
      id: `${id_music_prefix}.upload`,
      name: 'upload file',
      type: () => {
        return $el('button', {
          textContent: 'upload',
          onclick: () => {
            fileInput.click();
          },
        });
      },
    });

    app.ui.settings.addSetting({
      id: `${id_music_prefix}.fail_tip`,
      name: 'fail tip',
      tooltip: 'workflow fail  tip',
      type: 'boolean',
      defaultValue: false,
      onChange(v) {
        fail_tip = v;
      },
    });
  },
});
