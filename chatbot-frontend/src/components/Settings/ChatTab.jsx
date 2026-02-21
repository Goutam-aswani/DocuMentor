import { Select } from '../ui/Select';
import { ToggleSwitch } from '../ui/ToggleSwitch';

export const ChatTab = ({ settings, handleSettingChange }) => {
  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Language</label>
        <Select
          value={settings.language}
          onChange={(value) => handleSettingChange('language', value)}
          options={[
            { value: 'en', label: 'English' },
            { value: 'es', label: 'Spanish' },
            { value: 'hi', label: 'Hindi' },
            { value: 'fr', label: 'French' },
            { value: 'de', label: 'German' }
          ]}
        />
      </div>
      
      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Save Chat History</span>
          <p className="text-sm text-gray-400">Store your conversation history</p>
        </div>
        <ToggleSwitch
          checked={settings.saveHistory}
          onChange={(checked) => handleSettingChange('saveHistory', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Auto-save Conversations</span>
          <p className="text-sm text-gray-400">Automatically save ongoing chats</p>
        </div>
        <ToggleSwitch
          checked={settings.autoSave}
          onChange={(checked) => handleSettingChange('autoSave', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Message Sounds</span>
          <p className="text-sm text-gray-400">Play sound when receiving messages</p>
        </div>
        <ToggleSwitch
          checked={settings.messageSound}
          onChange={(checked) => handleSettingChange('messageSound', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Typing Indicator</span>
          <p className="text-sm text-gray-400">Show when you're typing</p>
        </div>
        <ToggleSwitch
          checked={settings.typingIndicator}
          onChange={(checked) => handleSettingChange('typingIndicator', checked)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Message Layout</label>
        <Select
          value={settings.messageLayout}
          onChange={(value) => handleSettingChange('messageLayout', value)}
          options={[
            { value: 'compact', label: 'Compact' },
            { value: 'comfortable', label: 'Comfortable' },
            { value: 'spacious', label: 'Spacious' }
          ]}
        />
      </div>
    </div>
  );
};