import { Select } from '../ui/Select';
import { ToggleSwitch } from '../ui/ToggleSwitch';

export const NotificationsTab = ({ settings, handleSettingChange }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Push Notifications</span>
          <p className="text-sm text-gray-400">Receive push notifications</p>
        </div>
        <ToggleSwitch
          checked={settings.pushNotifications}
          onChange={(checked) => handleSettingChange('pushNotifications', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Sound Notifications</span>
          <p className="text-sm text-gray-400">Play notification sounds</p>
        </div>
        <ToggleSwitch
          checked={settings.soundNotifications}
          onChange={(checked) => handleSettingChange('soundNotifications', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Desktop Notifications</span>
          <p className="text-sm text-gray-400">Show desktop notifications</p>
        </div>
        <ToggleSwitch
          checked={settings.desktopNotifications}
          onChange={(checked) => handleSettingChange('desktopNotifications', checked)}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Email Notifications</label>
        <Select
          value={settings.emailNotifications}
          onChange={(value) => handleSettingChange('emailNotifications', value)}
          options={[
            { value: 'all', label: 'All notifications' },
            { value: 'important', label: 'Important only' },
            { value: 'none', label: 'None' }
          ]}
        />
      </div>
    </div>
  );
};