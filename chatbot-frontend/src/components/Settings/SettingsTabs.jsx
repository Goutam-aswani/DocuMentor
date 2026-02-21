import {
  User,
  MessageSquare,
  Bell,
  Shield,
  Palette
} from 'lucide-react';

const tabs = [
  { id: 'account', label: 'Account', icon: User },
  { id: 'chat', label: 'Chat', icon: MessageSquare },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'privacy', label: 'Privacy', icon: Shield },
  { id: 'appearance', label: 'Appearance', icon: Palette }
];

export const SettingsTabs = ({ activeTab, setActiveTab }) => {
  return (
    <div className="lg:w-64 flex-shrink-0">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-4 space-y-2">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <Icon size={18} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};