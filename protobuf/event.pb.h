// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: event.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_event_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_event_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3021000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3021007 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_event_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_event_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_event_2eproto;
namespace naki3d {
namespace common {
namespace protocol {
class EventData;
struct EventDataDefaultTypeInternal;
extern EventDataDefaultTypeInternal _EventData_default_instance_;
}  // namespace protocol
}  // namespace common
}  // namespace naki3d
PROTOBUF_NAMESPACE_OPEN
template<> ::naki3d::common::protocol::EventData* Arena::CreateMaybeMessage<::naki3d::common::protocol::EventData>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace naki3d {
namespace common {
namespace protocol {

// ===================================================================

class EventData final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:naki3d.common.protocol.EventData) */ {
 public:
  inline EventData() : EventData(nullptr) {}
  ~EventData() override;
  explicit PROTOBUF_CONSTEXPR EventData(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  EventData(const EventData& from);
  EventData(EventData&& from) noexcept
    : EventData() {
    *this = ::std::move(from);
  }

  inline EventData& operator=(const EventData& from) {
    CopyFrom(from);
    return *this;
  }
  inline EventData& operator=(EventData&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const EventData& default_instance() {
    return *internal_default_instance();
  }
  static inline const EventData* internal_default_instance() {
    return reinterpret_cast<const EventData*>(
               &_EventData_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(EventData& a, EventData& b) {
    a.Swap(&b);
  }
  inline void Swap(EventData* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(EventData* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  EventData* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<EventData>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const EventData& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const EventData& from) {
    EventData::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(EventData* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "naki3d.common.protocol.EventData";
  }
  protected:
  explicit EventData(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kNameFieldNumber = 1,
    kParametersFieldNumber = 2,
  };
  // string name = 1;
  void clear_name();
  const std::string& name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_name();
  PROTOBUF_NODISCARD std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // string parameters = 2;
  void clear_parameters();
  const std::string& parameters() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_parameters(ArgT0&& arg0, ArgT... args);
  std::string* mutable_parameters();
  PROTOBUF_NODISCARD std::string* release_parameters();
  void set_allocated_parameters(std::string* parameters);
  private:
  const std::string& _internal_parameters() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_parameters(const std::string& value);
  std::string* _internal_mutable_parameters();
  public:

  // @@protoc_insertion_point(class_scope:naki3d.common.protocol.EventData)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr parameters_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_event_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// EventData

// string name = 1;
inline void EventData::clear_name() {
  _impl_.name_.ClearToEmpty();
}
inline const std::string& EventData::name() const {
  // @@protoc_insertion_point(field_get:naki3d.common.protocol.EventData.name)
  return _internal_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void EventData::set_name(ArgT0&& arg0, ArgT... args) {
 
 _impl_.name_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:naki3d.common.protocol.EventData.name)
}
inline std::string* EventData::mutable_name() {
  std::string* _s = _internal_mutable_name();
  // @@protoc_insertion_point(field_mutable:naki3d.common.protocol.EventData.name)
  return _s;
}
inline const std::string& EventData::_internal_name() const {
  return _impl_.name_.Get();
}
inline void EventData::_internal_set_name(const std::string& value) {
  
  _impl_.name_.Set(value, GetArenaForAllocation());
}
inline std::string* EventData::_internal_mutable_name() {
  
  return _impl_.name_.Mutable(GetArenaForAllocation());
}
inline std::string* EventData::release_name() {
  // @@protoc_insertion_point(field_release:naki3d.common.protocol.EventData.name)
  return _impl_.name_.Release();
}
inline void EventData::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  _impl_.name_.SetAllocated(name, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.name_.IsDefault()) {
    _impl_.name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:naki3d.common.protocol.EventData.name)
}

// string parameters = 2;
inline void EventData::clear_parameters() {
  _impl_.parameters_.ClearToEmpty();
}
inline const std::string& EventData::parameters() const {
  // @@protoc_insertion_point(field_get:naki3d.common.protocol.EventData.parameters)
  return _internal_parameters();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void EventData::set_parameters(ArgT0&& arg0, ArgT... args) {
 
 _impl_.parameters_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:naki3d.common.protocol.EventData.parameters)
}
inline std::string* EventData::mutable_parameters() {
  std::string* _s = _internal_mutable_parameters();
  // @@protoc_insertion_point(field_mutable:naki3d.common.protocol.EventData.parameters)
  return _s;
}
inline const std::string& EventData::_internal_parameters() const {
  return _impl_.parameters_.Get();
}
inline void EventData::_internal_set_parameters(const std::string& value) {
  
  _impl_.parameters_.Set(value, GetArenaForAllocation());
}
inline std::string* EventData::_internal_mutable_parameters() {
  
  return _impl_.parameters_.Mutable(GetArenaForAllocation());
}
inline std::string* EventData::release_parameters() {
  // @@protoc_insertion_point(field_release:naki3d.common.protocol.EventData.parameters)
  return _impl_.parameters_.Release();
}
inline void EventData::set_allocated_parameters(std::string* parameters) {
  if (parameters != nullptr) {
    
  } else {
    
  }
  _impl_.parameters_.SetAllocated(parameters, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.parameters_.IsDefault()) {
    _impl_.parameters_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:naki3d.common.protocol.EventData.parameters)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace protocol
}  // namespace common
}  // namespace naki3d

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_event_2eproto
