// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "MetadataView.h"

#include <vital/types/metadata_traits.h>

#include <vital/range/filter.h>
#include <vital/range/valid.h>
#include <vital/range/iota.h>

#include <qtColorUtil.h>
#include <qtSqueezedLabel.h>
#include <qtStlUtil.h>

#include <QEvent>
#include <QHash>
#include <QLabel>
#include <QScrollBar>
#include <QSet>
#include <QSpacerItem>
#include <QVariant>
#include <QVBoxLayout>

namespace kv = kwiver::vital;
namespace kvr = kwiver::vital::range;

namespace kwiver {
namespace arrows {
namespace qt {

QTE_IMPLEMENT_D_FUNC(MetadataView)

///////////////////////////////////////////////////////////////////////////////

//BEGIN MetadataViewPrivate

//-----------------------------------------------------------------------------
class MetadataViewPrivate
{
public:
  void addItem(int id, QString const& keyText);
  void setItemValue(int id, QString const& valueText);
  void clearItemValue(int id);

  void removeItem(int id);
  void clear();

  QSet<int> itemIds() const { return this->m_keyLabels.keys().toSet(); }
  void updateLabelColors();

  QWidget* m_contentWidget;

protected:
  static void setKeyLabelColor(QLabel* label);

  QHash<int, QLabel*> m_keyLabels;
  QHash<int, qtSqueezedLabel*> m_valueLabels;
};

//-----------------------------------------------------------------------------
void MetadataViewPrivate::addItem(int id, QString const& keyText)
{
  if (auto* const l = this->m_keyLabels.value(id, nullptr))
  {
    l->setText(keyText);
  }
  else
  {
    auto* const q = this->m_contentWidget;

    auto* const keyLabel = new QLabel{keyText, q};
    auto* const valueLabel = new qtSqueezedLabel{q};

    this->setKeyLabelColor(keyLabel);
    valueLabel->setTextMargins(1.0, 0.0);
    valueLabel->setElideMode(qtSqueezedLabel::ElideFade);

    auto* const layout = qobject_cast<QVBoxLayout*>(q->layout());
    if (layout->count())
    {
      delete layout->takeAt(layout->count() - 1);
    }
    layout->addWidget(keyLabel);
    layout->addWidget(valueLabel);
    layout->addStretch(1);

    keyLabel->show();
    valueLabel->show();

    q->resize(q->sizeHint());

    this->m_keyLabels.insert(id, keyLabel);
    this->m_valueLabels.insert(id, valueLabel);
    this->clearItemValue(id);
  }
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::setItemValue(int id, QString const& valueText)
{
  if (auto* const l = this->m_valueLabels.value(id, nullptr))
  {
    if (valueText.isEmpty())
    {
      l->setText("(empty)");
      l->setToolTip({});
      l->setEnabled(false);
    }
    else
    {
      l->setText(valueText.trimmed(), qtSqueezedLabel::SetToolTip);
      l->setEnabled(true);
    }
  }
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::clearItemValue(int id)
{
  if (auto* const l = this->m_valueLabels.value(id, nullptr))
  {
    l->setText("(not available)");
    l->setToolTip({});
    l->setEnabled(false);
  }
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::removeItem(int id)
{
  delete this->m_keyLabels.take(id);
  delete this->m_valueLabels.take(id);
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::clear()
{
  qDeleteAll(this->m_keyLabels);
  qDeleteAll(this->m_valueLabels);
  this->m_keyLabels.clear();
  this->m_valueLabels.clear();
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::setKeyLabelColor(QLabel* label)
{
  auto const& p = label->palette();
  auto const& bg = p.color(QPalette::Window);
  auto const& fg = p.color(QPalette::WindowText);
  auto const& c = qtColorUtil::blend(bg, fg, 0.6);
  label->setStyleSheet(QString{"color: %1;"}.arg(c.name()));
}

//-----------------------------------------------------------------------------
void MetadataViewPrivate::updateLabelColors()
{
  for (auto* const l : this->m_keyLabels)
  {
    this->setKeyLabelColor(l);
  }
}

//END MetadataViewPrivate

///////////////////////////////////////////////////////////////////////////////

//BEGIN MetadataView

//-----------------------------------------------------------------------------
MetadataView::MetadataView(QWidget* parent)
  : QScrollArea{parent}, d_ptr{new MetadataViewPrivate}
{
  QTE_D();

  d->m_contentWidget = new QWidget{this};
  d->m_contentWidget->setLayout(new QVBoxLayout);
  d->m_contentWidget->installEventFilter(this);

  this->setWidget(d->m_contentWidget);
}

//-----------------------------------------------------------------------------
MetadataView::~MetadataView()
{
}

//-----------------------------------------------------------------------------
bool MetadataView::eventFilter(QObject* sender, QEvent* e)
{
  QTE_D();

  if (sender == d->m_contentWidget && e && e->type() == QEvent::Resize)
  {
    this->setMinimumWidth(
      d->m_contentWidget->minimumSizeHint().width() +
      this->verticalScrollBar()->width());
  }

  return QScrollArea::eventFilter(sender, e);
}

//-----------------------------------------------------------------------------
void MetadataView::changeEvent(QEvent* e)
{
  if (e && e->type() == QEvent::PaletteChange)
  {
    QTE_D();
    d->updateLabelColors();
  }

  QScrollArea::changeEvent(e);
}

//-----------------------------------------------------------------------------
void MetadataView::updateMetadata(
  std::shared_ptr<kv::metadata_map::map_metadata_t> mdMap)
{
  QTE_D();

  // Reset UI so fields will be in correct order
  d->clear();

  QSet<kv::vital_metadata_tag> mdKeys;
  auto traits = kv::metadata_traits{};

  // Collect all keys present in the metadata map
  for (auto const& mdi : *mdMap)
  {
    for (auto const& mdp : mdi.second | kvr::valid)
    {
      for (auto const& mde : *mdp)
      {
        mdKeys.insert(mde.first);
      }
    }
  }

  // Update UI fields
  using md_tag_type_t = std::underlying_type<kv::vital_metadata_tag>::type;
  constexpr auto lastMetadataTag =
    static_cast<md_tag_type_t>(kv::VITAL_META_LAST_TAG);
  for (auto const k : vital::range::iota(lastMetadataTag))
  {
    auto const tag = static_cast<kv::vital_metadata_tag>(k);
    if (mdKeys.contains(tag))
    {
      d->clearItemValue(k);
      d->addItem(k, qtString(traits.tag_to_name(tag)));
    }
  }
}

//-----------------------------------------------------------------------------
void MetadataView::updateMetadata(kv::metadata_vector const& mdVec)
{
  QTE_D();

  auto traits = kv::metadata_traits{};

  auto const keys = d->itemIds();
  for (auto const k : keys)
  {
    d->clearItemValue(k);
  }

  for (auto const& mdp : mdVec | kvr::valid)
  {
    for (auto const& mde : *mdp)
    {
      using md_tag_type_t = std::underlying_type<kv::vital_metadata_tag>::type;
      auto const k = static_cast<md_tag_type_t>(mde.first);
      auto const* mdi = mde.second.get();

      if (mdi)
      {
        if (!keys.contains(k))
        {
          auto const tag = static_cast<kv::vital_metadata_tag>(k);
          d->addItem(k, qtString(traits.tag_to_name(tag)));
        }
        d->setItemValue(k, qtString(mdi->as_string()));
      }
    }

    // TODO handle multiple metadatas?
    break;
  }
}

//END MetadataView

} // namespace qt
} // namespace arrows
} // namespace kwiver
